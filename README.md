# Dynamic TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series Forecasting

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-ICML%202026-red.svg)](https://icml.cc/virtual/2026/poster/64808)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-green.svg)](https://pytorch.org/)


</div>

This repository contains the official implementation of **Dynamic TMoE**, accepted as a poster at **ICML 2026**.

![framework](assets/framework.png)

<p align="center">
  <strong>Dynamic TMoE framework architecture</strong>
</p>

## 📖 Overview

Dynamic TMoE is an adaptive Mixture of Experts (MoE) framework for non-stationary time series forecasting. It overcomes the rigidity of traditional MoEs by using Maximum Mean Discrepancy (MMD) to detect distribution shifts, dynamically expanding or pruning its heterogeneous expert pool to optimize capacity. Paired with a temporal memory router for stable, context-aware expert selection, Dynamic TMoE achieves state-of-the-art results across nine benchmarks, outperforming advanced baselines by reducing MSE and MAE by an average of 10.4% and 7.8%, respectively.

## 🚀 Quick Start

### Environment Setup

To set up the environment, install Python 3.10 and the required dependencies:

```bash
conda create -n dynamic_tmoe python=3.10
conda activate dynamic_tmoe
pip install -r requirements.txt
```

### Dataset Preparation

Place the benchmark datasets in the `./dataset` folder with the following structure:

```text
dataset/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── electricity/
│   └── electricity.csv
├── traffic/
│   └── traffic.csv
├── weather/
│   └── weather.csv
├── exchange_rate/
│   └── exchange_rate.csv
└── illness/
    └── national_illness.csv
```

### Running Experiments

Run the following scripts for different long-term forecasting benchmarks:

```bash
# eg. ETTh1 benchmarks
bash ./scripts/ETTh1.sh

```

## 📚 Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{
  zhu2026dynamic_tmoe,
  title={Dynamic {TMoE}: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series Forecasting},
  author={Jiawen Zhu and Shuhan Liu and Di Weng and Yingcai Wu},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=JabkBcaoa9}
}
```

## 🙏 Acknowledgement
We thanks to the following repositories for their invaluable code and datasets:

- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
