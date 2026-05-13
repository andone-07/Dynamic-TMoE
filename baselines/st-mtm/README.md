
ST-MTM (KDD 2025)
===============

### Hyunwoo Seo, Chiehyeon Lim


### This repository provides the official implementation of ST-MTM from the paper [ST-MTM: Masked Time Series Modeling with Seasonal-Trend Decomposition for Time Series Forecasting](https://dl.acm.org/doi/10.1145/3690624.3709254).


# Requirements

- Python 3.9.0
- torch==2.0.1
- numpy==1.24.3
- pandas==1.5.3
- scikit-learn==1.2.2
- matplotlib==3.7.1
- tensorboardX==2.6.2.2

Dependencies can be installed using the following command:

    pip install -r requirements.txt

# Getting Started

## 1. Prepare Data

All benchmark datasets can be obtained from [Google Drive](https://drive.google.com/drive/folders/1lXyUYd0RerPyJgUZZ5rv0uYG-xcycQ38?usp=sharing), and arrange the folder as:

    ST-MTM/
    |-- datasets/
        |-- ETTh1.csv
        |-- ETTh2.csv
        |-- ETTm1.csv
        |-- ETTm2.csv
        |-- Weather.csv
        |-- Electricity.csv
        |-- Exchange.csv
        |-- national_illness.csv
        |-- CWRU/
            |-- CWRU_Time_Normal_1_098_ds10.csv
        |-- solar_AL.txt
        |-- PEMS08/
            |-- PEMS08.npz

## 2. Experimental reproduction

- We provide the scripts for pre-training and finetuning for each dataset with the best hyper-parameters in our experiment at `./scripts/`.

### 2-1. Pre-training

Pre-training ST-MTM for each dataset can be implemented through the provided scripts in `./scripts/pretrain/`. For example, to pre-train ST-MTM for the ETTh1 dataset:

    bash scripts/pretrain/ETTh1.sh

### 2-2. Fine-tuning

After pre-training ST-MTM for the dataset, fine-tuning ST-MTM for forecasting across various lengths can be implemented through the provided scripts in `./scripts/finetune/`. For example, to fine-tune ST-MTM for the ETTh1 dataset:

    bash scripts/finetune/ETTh1.sh

### 2-3. Pre-training and fine-tuning at once

To implement pre-training and fine-tuning sequentially, the scripts in `./scripts/`. For example, to perform both steps at once for the electricity dataset:

    bash scripts/run_electricity.sh

For CWRU bearing forecasting:

    bash scripts/run_cwru.sh

# Complete Results of Multivariate Time Series Forecasting

We present the complete experimental results for multivariate time series forecasting, including performance across all prediction lengths.

![self-supervised baselines](img/self-supervised%20learning.png)
![decomposition-based forecasting baselines](img/decomposition-based%20forecasting.png)

# Contact

If you have any questions or concerns. please contact ta57xr@unist.ac.kr or submit an issue.

# Citation

If you find this repo useful in your research, please consider citing our paper as follows:

