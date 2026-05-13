from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, PatchTST_MoE_cluster
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from sklearn.cluster import KMeans

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

from thop import profile

from layers.Cluster import EDESC
from layers.InitializeD import Initialization_D
from layers.RevIN import RevIN

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'PatchTST_MoE_cluster': PatchTST_MoE_cluster,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _get_profile(self, model):
        _input=torch.randn(self.args.batch_size, self.args.seq_len, self.args.enc_in).to(self.device)
        macs, params = profile(model, inputs=(_input,))
        print('FLOPs: ', macs)
        print('params: ', params)
        return macs, params

    def _refined_subspace_affinity(self, s):
        weight = s ** 2 / s.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        s_time, s_frequency, outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Update refined subspace affinity
                tmp_s_time = s_time.data
                s_tilde_time = self._refined_subspace_affinity(s=tmp_s_time)
                tmp_s_frequency = s_frequency.data
                s_tilde_frequency = self._refined_subspace_affinity(s=tmp_s_frequency)

                # Total loss function
                n_z = self.args.c_out * self.args.d_model
                T_dim = int(n_z / self.args.T_num_expert)
                F_dim = int(n_z / self.args.F_num_expert)
                loss_cluster_time = self.model.model_time.cluster.total_loss(pred=s_time, target=s_tilde_time,
                                                                        dim=T_dim, n_clusters=self.args.T_num_expert,
                                                                        beta=self.args.beta)
                loss_cluster_frequency = self.model.model_frequency.cluster.total_loss(pred=s_frequency, target=s_tilde_frequency,
                                                                             dim=F_dim, n_clusters=self.args.F_num_expert,
                                                                             beta=self.args.beta)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true) + self.args.alpha * loss_cluster_time + self.args.gama * loss_cluster_frequency

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        print(self.model)
        self._get_profile(self.model)
        print('Trainable parameters: ', sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        training_start_time = time.time()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            s_time, s_frequency, outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    # Update refined subspace affinity
                    tmp_s_time = s_time.data
                    s_tilde_time = self._refined_subspace_affinity(s=tmp_s_time)
                    tmp_s_frequency = s_frequency.data
                    s_tilde_frequency = self._refined_subspace_affinity(s=tmp_s_frequency)

                    # Total loss function
                    n_z = self.args.c_out * self.args.d_model
                    T_dim = int(n_z / self.args.T_num_expert)
                    F_dim = int(n_z / self.args.F_num_expert)
                    loss_cluster_time = self.model.model_time.cluster.total_loss(pred=s_time, target=s_tilde_time,
                                                                       dim=T_dim, n_clusters=self.args.T_num_expert,
                                                                       beta=self.args.beta)
                    loss_cluster_frequency = self.model.model_frequency.cluster.total_loss(pred=s_frequency, target=s_tilde_frequency,
                                                                       dim=F_dim, n_clusters=self.args.F_num_expert,
                                                                       beta=self.args.beta)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss_fore = criterion(outputs, batch_y)
                    loss = loss_fore + self.args.alpha * loss_cluster_time + self.args.gama * loss_cluster_frequency

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # Save training time log
        total_training_time = time.time() - training_start_time
        training_time_log_folder = './training_time_logs/'
        if not os.path.exists(training_time_log_folder):
            os.makedirs(training_time_log_folder)

        model_id_parts = self.args.model_id.split('_')
        if len(model_id_parts) >= 3:
            dataset_name = model_id_parts[0]
            lookback = model_id_parts[1]
            predict = model_id_parts[2]
            time_log_filename = f'training_time_{dataset_name}_{lookback}_{predict}.txt'
        else:
            time_log_filename = f'training_time_{self.args.model_id}.txt'

        time_log_path = os.path.join(training_time_log_folder, time_log_filename)
        num_epochs_trained = epoch + 1
        with open(time_log_path, 'w') as f:
            f.write(f"Model: {self.args.model}\n")
            f.write(f"Setting: {setting}\n")
            f.write(f"Epochs Trained: {num_epochs_trained}\n")
            f.write(f"Total Training Time: {total_training_time:.4f} s\n")
            f.write(f"Average Time per Epoch: {total_training_time / num_epochs_trained:.4f} s\n")

        print(f"\nTraining Time Summary:")
        print(f"Total Training Time: {total_training_time:.4f} s")
        print(f"Epochs Trained: {num_epochs_trained}")
        print(f"Average Time per Epoch: {total_training_time / num_epochs_trained:.4f} s")
        print(f"Training time log saved to: {time_log_path}")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        # Track GPU memory usage for reporting in logs
        gpu_device = None
        base_allocated = 0
        if self.args.use_gpu and torch.cuda.is_available():
            gpu_device = torch.device(f'cuda:{self.args.gpu}')
            torch.cuda.synchronize(gpu_device)
            torch.cuda.reset_peak_memory_stats(gpu_device)
            base_allocated = torch.cuda.memory_allocated(gpu_device)
        
        if test:
            print('loading model')
            checkpoint = torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            # 过滤掉total_ops和total_params相关的键
            checkpoint = {k: v for k, v in checkpoint.items() if 'total_ops' not in k and 'total_params' not in k}
            self.model.load_state_dict(checkpoint, strict=False)
        # 计算模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # total_params = sum(p.numel() for p in self.model.parameters())

        print("模型总参数数量:", total_params)

        preds = []
        trues = []
        clusters_time = []
        clusters_frequency = []
        inputx = []
        inference_times = []  # Track all inference times
        gpu_memory_used = []  # Track GPU memory usage
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Create performance logs folder
        performance_folder = './performance_logs/'
        if not os.path.exists(performance_folder):
            os.makedirs(performance_folder)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                
                # Measure inference time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.time()  # 计时开始
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            s_time, s_frequency, outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_times.append(time.time() - start_time)  # 记录推理时间
                
                # Record GPU memory
                if torch.cuda.is_available():
                    gpu_memory_used.append(torch.cuda.memory_allocated(self.device) / 1024**2)
                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                cluster_time = s_time.detach().cpu().numpy()
                cluster_frequency = s_frequency.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                cluster_time = cluster_time
                cluster_frequency = cluster_frequency

                preds.append(pred)
                trues.append(true)
                clusters_time.append(cluster_time)
                clusters_frequency.append(cluster_frequency)

                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # Report GPU memory usage (MB) if GPU is available
        if gpu_device is not None:
            torch.cuda.synchronize(gpu_device)
            current_alloc = torch.cuda.memory_allocated(gpu_device)
            peak_alloc = torch.cuda.max_memory_allocated(gpu_device)
            inference_used = max(0, peak_alloc - base_allocated)
            to_mb = lambda x: x / (1024 ** 2)
            print('[GPU Memory Usage]')
            print('  Current Memory Allocated: {:.2f} MB'.format(to_mb(current_alloc)))
            print('  Peak Memory Allocated:    {:.2f} MB'.format(to_mb(peak_alloc)))
            print('  Memory Used (inference):  {:.2f} MB'.format(to_mb(inference_used)))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        clusters_time = np.array(clusters_time)
        clusters_frequency = np.array(clusters_frequency)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # Calculate performance metrics
        avg_inference_time_ms = np.mean(inference_times) * 1000
        avg_gpu_memory = np.mean(gpu_memory_used) if gpu_memory_used else 0
        
        # Calculate per-sample metrics
        total_samples = preds.shape[0]
        batch_size = self.args.batch_size
        inference_time_per_sample = avg_inference_time_ms / batch_size
        throughput = 1000.0 / inference_time_per_sample  # samples per second

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        result_file = './result_{}_{}.txt'.format(self.args.data_path.split('.')[0], self.args.seq_len)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        
        # Print metrics
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        print('Total Params: {:,}'.format(total_params))
        print('Batch Size: {0}, Total Samples: {1}'.format(batch_size, total_samples))
        print('Avg Inference Time: {0:.4f} ms/batch'.format(avg_inference_time_ms))
        print('Inference Time per Sample: {0:.4f} ms/sample'.format(inference_time_per_sample))
        print('Throughput: {0:.2f} samples/sec'.format(throughput))
        if torch.cuda.is_available():
            print('Avg GPU Memory: {0:.2f} MB'.format(avg_gpu_memory))
        
        f = open(result_file, 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()
        
        # Write performance log with new naming format
        performance_filename = 'performance_{}_{}_{}.log'.format(
            self.args.data_path.split('.')[0], 
            self.args.seq_len, 
            self.args.pred_len
        )
        log_file = open(performance_folder + performance_filename, 'w')
        log_file.write('Model: {}\n'.format(self.args.model))
        log_file.write('Dataset: {}\n'.format(self.args.data_path.split('.')[0]))
        log_file.write('Total Parameters: {:,}\n'.format(total_params))
        log_file.write('Trainable Parameters: {:,}\n'.format(total_params))
        log_file.write('Batch Size: {}\n'.format(batch_size))
        log_file.write('Total Samples: {}\n'.format(total_samples))
        log_file.write('Average Inference Time: {:.4f} ms/batch\n'.format(avg_inference_time_ms))
        log_file.write('Inference Time per Sample: {:.4f} ms/sample\n'.format(inference_time_per_sample))
        log_file.write('Throughput: {:.2f} samples/sec\n'.format(throughput))
        if torch.cuda.is_available():
            log_file.write('Average GPU Memory: {:.2f} MB\n'.format(avg_gpu_memory))
        log_file.write('MSE: {:.6f}\n'.format(mse))
        log_file.write('MAE: {:.6f}\n'.format(mae))
        log_file.write('RSE: {:.6f}\n'.format(rse))
        log_file.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'cluster_time_result.npy', clusters_time)
        np.save(folder_path + 'cluster_frequency_result.npy', clusters_frequency)
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
