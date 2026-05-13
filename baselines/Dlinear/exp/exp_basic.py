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
        if self.args.use_gpu:
            available = torch.cuda.device_count()
            if available == 0:
                self.args.use_gpu = False
            else:
                if not self.args.use_multi_gpu:
                    if self.args.gpu >= available:
                        # clamp to valid single device index
                        self.args.gpu = available - 1
                    device = torch.device(f'cuda:{self.args.gpu}')
                    print(f'Use GPU: cuda:{self.args.gpu}')
                    return device
                else:
                    # multi-gpu: rely on global device ids provided
                    device_ids = [int(x) for x in self.args.devices.split(',') if x.strip()]
                    if len(device_ids) == 0:
                        self.args.use_gpu = False
                    else:
                        self.args.device_ids = device_ids
                        # ensure primary gpu is valid
                        primary = device_ids[0] if device_ids[0] < available else 0
                        self.args.gpu = primary
                        device = torch.device(f'cuda:{primary}')
                        print(f'Use GPUs: {device_ids}, primary cuda:{primary}')
                        return device
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
