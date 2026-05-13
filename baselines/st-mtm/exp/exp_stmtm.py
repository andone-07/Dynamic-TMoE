from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, transfer_weights, show_series, show_matrix, visual
# from utils.augmentations import masked_data
from utils.metrics import metric
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from collections import OrderedDict
from tensorboardX import SummaryWriter
import random

warnings.filterwarnings('ignore')


class Exp_STMTM(Exp_Basic):
    def __init__(self, args):
        super(Exp_STMTM, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs")

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.load_checkpoints:
            print("Loading ckpt: {}".format(self.args.load_checkpoints))
            print(self.device)
            print('torch.cuda.device_count()', torch.cuda.device_count())
            model = transfer_weights(self.args.load_checkpoints, model, device=self.device)

        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!", self.args.device_ids)
        #     model = nn.DataParallel(model, device_ids=self.args.device_ids)
        # print out the model size
        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))

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

    def pretrain(self):

        # data preparation
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        # show cases
        self.train_show = next(iter(train_loader))
        self.valid_show = next(iter(vali_loader))

        path = os.path.join(self.args.pretrain_checkpoints, self.args.data)
        if not os.path.exists(path):
            os.makedirs(path)

        # optimizer
        model_optim = self._select_optimizer()
        #model_optim.add_param_group({'params': self.awl.parameters(), 'weight_decay': 0})
        model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optim,
                                                                     T_max=self.args.train_epochs)

        # pre-training
        min_vali_loss = None
        for epoch in range(self.args.train_epochs):
            start_time = time.time()

            # train_loss = self.pretrain_one_epoch(train_loader, model_optim, model_scheduler)
            # vali_loss = self.valid_one_epoch(vali_loader)
            train_loss, train_mse_loss, train_cont_loss = self.pretrain_one_epoch(train_loader, model_optim, model_scheduler)
            vali_loss, vali_mse_loss, vali_cont_loss = self.valid_one_epoch(vali_loader)
            

            # log and Loss
            end_time = time.time()
            print(
                "Epoch: {0}, Lr: {1:.7f}, Time: {2:.2f}s | Train Loss: {3:.4f} ({4:.3f}, {5:.3f}) Val Loss: {6:.4f} ({7:.3f}, {8:.3f})"
                .format(epoch, model_scheduler.get_lr()[0], end_time - start_time, 
                        train_loss, train_mse_loss, train_cont_loss, 
                        vali_loss, vali_mse_loss, vali_cont_loss))

            loss_scalar_dict = {
                'train_loss': train_loss,
                'train_mse_loss' : train_mse_loss,
                'train_cont_loss' : train_cont_loss,
                'vali_loss': vali_loss,
                'vali_mse_loss': vali_mse_loss,
                'vali_cont_loss': vali_cont_loss
            }

            self.writer.add_scalars(f"/pretrain_loss", loss_scalar_dict, epoch)

            # checkpoint saving
            if not min_vali_loss or vali_loss <= min_vali_loss:
                if epoch == 0:
                    min_vali_loss = vali_loss

                print(
                    "Validation loss decreased ({0:.4f} --> {1:.4f}).  Saving model epoch{2} ...".format(min_vali_loss, vali_loss, epoch))

                min_vali_loss = vali_loss
                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if 'encoder' in k or 'embedding' in k:
                        if 'module.' in k:
                            k = k.replace('module.', '')  # multi-gpu
                        self.encoder_state_dict[k] = v
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.encoder_state_dict}
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt_best.pth"))

            if (epoch + 1) % 10 == 0:
                print("Saving model at epoch {}...".format(epoch + 1))

                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if 'encoder' in k or 'embedding' in k:
                        if 'module.' in k:
                            k = k.replace('module.', '')
                        self.encoder_state_dict[k] = v
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.encoder_state_dict}
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt{epoch + 1}.pth"))


    def pretrain_one_epoch(self, train_loader, model_optim, model_scheduler):
        
        train_loss = []
        train_cl_loss = []
        train_mse_loss = []

        self.model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            model_optim.zero_grad()

            # if self.args.select_channels < 1:

            #     # random select channels
            #     B, S, C = batch_x.shape
            #     random_c = int(C * self.args.select_channels)
            #     if random_c < 1:
            #         random_c = 1

            #     index = torch.LongTensor(random.sample(range(C), random_c))
            #     batch_x = torch.index_select(batch_x, 2, index)

            # to device
            batch_x = batch_x.float().to(self.device)
            # encoder
            loss, mse_loss, cont_loss = self.model(batch_x, batch_x_mark)
            
            # backward
            loss.backward()
            model_optim.step()
            
            # record
            train_loss.append(loss.item())
            train_mse_loss.append(mse_loss.item())
            train_cl_loss.append(cont_loss.item())

        model_scheduler.step()
        
        train_loss = np.average(train_loss)
        train_mse_loss = np.average(train_mse_loss)
        train_cl_loss = np.average(train_cl_loss)

        return train_loss, train_mse_loss, train_cl_loss

    def valid_one_epoch(self, vali_loader):
        valid_loss = []
        valid_cl_loss = []
        valid_mse_loss = []

        self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            # to device
            batch_x = batch_x.float().to(self.device)
            loss, mse_loss, cont_loss = self.model(batch_x, batch_x_mark)
            
            # Record
            valid_loss.append(loss.item())
            valid_mse_loss.append(mse_loss.item())
            valid_cl_loss.append(cont_loss.item())

        valid_loss = np.average(valid_loss)
        valid_mse_loss = np.average(valid_mse_loss)
        valid_cl_loss = np.average(valid_cl_loss)

        self.model.train()
        return valid_loss, valid_mse_loss, valid_cl_loss
    
    def train(self, setting):

        # data preparation
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # Optimizer
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)    
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            start_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                # to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # encoder
                outputs = self.model(batch_x, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # loss
                loss = criterion(outputs, batch_y)
                loss.backward()
                model_optim.step()

                # record
                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)

            end_time = time.time()
            print(
            "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f}".format(
                epoch + 1, train_steps, end_time - start_time, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        self.lr = model_optim.param_groups[0]['lr']

        return self.model

    def vali(self, vali_loader, criterion):
        total_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # encoder
                outputs = self.model(batch_x, batch_x_mark)

                # loss
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)

                # record
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Create performance logs folder
        performance_folder = './performance_logs/'
        if not os.path.exists(performance_folder):
            os.makedirs(performance_folder)

        # Calculate model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Track inference time and GPU memory
        inference_times = []
        gpu_memory_used = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # Measure inference time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.time()
                
                outputs = self.model(batch_x, batch_x_mark)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_times.append(time.time() - start_time)
                
                # Record GPU memory
                if torch.cuda.is_available():
                    gpu_memory_used.append(torch.cuda.memory_allocated(self.device) / 1024**2)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        
        # Calculate averages (convert to milliseconds)
        avg_inference_time_ms = np.mean(inference_times) * 1000
        avg_gpu_memory = np.mean(gpu_memory_used) if gpu_memory_used else 0
        
        # Calculate per-sample metrics
        total_samples = preds.shape[0]
        batch_size = self.args.batch_size
        inference_time_per_sample = avg_inference_time_ms / batch_size
        throughput = 1000.0 / inference_time_per_sample  # samples per second
        
        # Print metrics
        print('{0}->{1}, mse:{2:.3f}, mae:{3:.3f}'.format(self.args.seq_len, self.args.pred_len, mse, mae))
        print('Total Params: {0:,}, Trainable Params: {1:,}'.format(total_params, trainable_params))
        print('Batch Size: {0}, Total Samples: {1}'.format(batch_size, total_samples))
        print('Avg Inference Time: {0:.4f} ms/batch'.format(avg_inference_time_ms))
        print('Inference Time per Sample: {0:.4f} ms/sample'.format(inference_time_per_sample))
        print('Throughput: {0:.2f} samples/sec'.format(throughput))
        if torch.cuda.is_available():
            print('Avg GPU Memory: {0:.2f} MB'.format(avg_gpu_memory))
        
        # Write to score file
        f = open(folder_path + "score.txt", 'a')
        if self.args.pred_len==96:
            f.write(setting+'\n')
        f.write('{0}->{1}, {2:.3f}, {3:.3f} \n'.format(self.args.seq_len, self.args.pred_len, mse, mae))
        f.close()
        
        # Write performance log with new naming format
        performance_filename = 'performance_{}_{}_{}.log'.format(
            self.args.data, 
            self.args.seq_len, 
            self.args.pred_len
        )
        log_file = open(performance_folder + performance_filename, 'w')
        log_file.write('Model: {}\n'.format(self.args.model))
        log_file.write('Dataset: {}\n'.format(self.args.data))
        log_file.write('Total Parameters: {:,}\n'.format(total_params))
        log_file.write('Trainable Parameters: {:,}\n'.format(trainable_params))
        log_file.write('Batch Size: {}\n'.format(batch_size))
        log_file.write('Total Samples: {}\n'.format(total_samples))
        log_file.write('Average Inference Time: {:.4f} ms/batch\n'.format(avg_inference_time_ms))
        log_file.write('Inference Time per Sample: {:.4f} ms/sample\n'.format(inference_time_per_sample))
        log_file.write('Throughput: {:.2f} samples/sec\n'.format(throughput))
        if torch.cuda.is_available():
            log_file.write('Average GPU Memory: {:.2f} MB\n'.format(avg_gpu_memory))
        log_file.write('MSE: {:.6f}\n'.format(mse))
        log_file.write('MAE: {:.6f}\n'.format(mae))
        log_file.close()