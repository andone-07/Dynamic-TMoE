import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            visible_gpus = torch.cuda.device_count()
            if visible_gpus == 0:
                device = torch.device('cpu')
                print('Use CPU (no visible GPUs)')
                return device

            if self.args.gpu >= visible_gpus:
                print('Requested gpu id {} but only {} visible; falling back to 0'.format(
                    self.args.gpu, visible_gpus))
                self.args.gpu = 0

            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
