from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_PEMS, Dataset_Solar
from torch.utils.data import DataLoader

import torch
import numpy as np

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Weather': Dataset_Custom,
    'Electricity': Dataset_Custom,
    'Exchange' : Dataset_Custom,
    'PEMS08' : Dataset_PEMS,
    'ILI': Dataset_Custom,
    'Solar': Dataset_Solar,
    'Traffic': Dataset_Custom,
    'CWRU': Dataset_Custom,
    'MIT-BIH': Dataset_Custom,
}
    
def data_provider(args, flag):
    Data = data_dict[args.data]

    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size  # Use same batch_size as other models for consistent data sampling
        freq = args.freq
    else:
        shuffle_flag = True if flag=='train' else False
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq


    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    
    print(flag, len(data_set), len(data_loader))
    return data_set, data_loader


