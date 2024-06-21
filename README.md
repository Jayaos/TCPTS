## Transformer Conformal Prediction for Time Series

This repository contains codes for implementing models and reproducing the results in our paper ["Transformer Conformal Prediction for Time Series"](https://arxiv.org/abs/2406.05332). Some parts of the codes are inherited from ["Sequential Predictive Conformal Inference for Time Series"](https://proceedings.mlr.press/v202/xu23r/xu23r.pdf).

    Transformer Conformal Prediction for Time Series
        Junghwan Lee, Chen Xu, and Yao Xie
        ICML 2024 Workshop on Structured Probabilistic Inference & Generative Modeling
        https://arxiv.org/abs/2406.05332

    Sequential Predictive Conformal Inference for Time Series
        Chen Xu and Yao Xie
        ICML 2023
        https://proceedings.mlr.press/v202/xu23r/xu23r.pdf

### Installation
The codes were written in Python 3.9.13. If you want to implement locally,

1. Clone this repository
2. Move to the directory where you clone this repository and install requirements using pip

        pip install -r requirements.txt

### Experiments using Colab Notebook
We also provide colab notebook for reproducing the results. Note that results from colab notebook shows slightly different performance from the results in the paper.

#### Simulation: time series with heteroscedastic errors
<a target="_blank" href="https://colab.research.google.com/github/Jayaos/TCPTS/blob/main/examples/hetero_example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Conformal prediction on time series with heteroscedastic errors as described in Section 4.2 in the paper.

#### Real Data Experiment: Electricity dataset
<a target="_blank" href="https://colab.research.google.com/github/Jayaos/TCPTS/blob/main/examples/electricity_example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Conformal prediction on the electricity dataset in Section 4.3 in the paper.