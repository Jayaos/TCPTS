import torch
from torch import nn, Tensor
import math
from torch.utils.data import Dataset
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view


def build_strided_residue_QuantileGPT(residue_sequence, window_len):
    
    strided_residue_sequences = sliding_window_view(residue_sequence, window_len)
    strided_residue_input_sequences = strided_residue_sequences[:-1]
    strided_residue_prediction_sequences = strided_residue_sequences[1:]
    
    return strided_residue_input_sequences, strided_residue_prediction_sequences

def build_strided_feature_QuantileGPT(feature_sequence, window_len):
    
    strided_feature_sequences = sliding_window_view(feature_sequence, window_len, axis=0)
    strided_feature_input_sequences = strided_feature_sequences[:-1]
    strided_feature_input_sequences = np.transpose(strided_feature_input_sequences, (0,2,1))
    
    return strided_feature_input_sequences

class DatasetQuantileGPT(Dataset):
    
    def __init__(self, residX, featureX, residY):
        self.residX = residX
        self.featureX = featureX
        self.residY = residY

    def __len__(self):
        return len(self.residY)

    def __getitem__(self,idx):
        return self.residX[idx,:], self.featureX[idx,:], self.residY[idx]
    
def collate_fn_QuantileGPT(batch):

    resid_x_batch, feature_x_batch, resid_y_batch = zip(*batch)

    resid_x_tensor = torch.Tensor(resid_x_batch) # batch_size * past_window
    feature_x_tensor = torch.Tensor(feature_x_batch) # batch_size * past_window * feature_dim 
    resid_y_tensor = torch.Tensor(resid_y_batch) # batch_size * num_step_prediction

    batch_size, past_window, feature_dim = feature_x_tensor.size()
    resid_x_tensor = resid_x_tensor.reshape((batch_size, past_window, 1)) # batch_size * past_window * 1
    x_tensor = torch.cat([resid_x_tensor, feature_x_tensor], axis=-1) # batch_size * past_window * (feature_dim+1)

    return x_tensor, resid_y_tensor

class QuantileGPT(nn.Module):

    """
    stacked Transformer Decoder for quantile regression
    Input: sequential features for endoder padded by max_seq_len and sequential features decoder
        sequential features: (batch_size * past_window * feature_dim)
    Output: predicted values for pre-defined quantiles (batch_size * step_prediction * # of pre-defined quantiles)
    """
    
    def __init__(self, dim_feature: int, dim_model: int, num_head: int, dim_ff: int, num_layer: int, 
                 target_quantiles: list, dropout: float = 0.1, batch_first: bool=True):
        super(QuantileGPT, self).__init__()
        self.dim_model = dim_model
        self.target_quantiles = target_quantiles
        self.positional_encoding = PositionalEncoding(dim_model, dropout)

        Encoder_Layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_head, dim_feedforward=dim_ff, 
                                                   dropout=dropout, batch_first=batch_first)
        self.Encoder = nn.TransformerEncoder(Encoder_Layer, num_layers=num_layer)

        self.input_linear = nn.Linear(dim_feature, dim_model) # this will work as embedding layer for features
        self.output_linear = nn.Linear(dim_model, len(target_quantiles)) # no activation

    def forward(self, src: Tensor, src_mask: Tensor, src_key_padding_mask: Tensor):

        src_emb = self.input_linear(src)
        src_emb = src_emb * math.sqrt(self.dim_model)
        src_emb = self.positional_encoding(src_emb)

        outputs = self.Encoder(src_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        return self.output_linear(outputs)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
def compute_CI_QGPT(output_list, past_window, upper_quantile_idx, lower_quantile_idx):
    
    CI_length = len(output_list)
    
    quantile_arr = np.array(output_list).reshape((CI_length, past_window, -1)) 
    quantile_arr = quantile_arr[:,-1,:] # CI_length * target_quantiles
    
    upper_quantile_arr = quantile_arr[:,upper_quantile_idx]
    lower_quantile_arr = quantile_arr[:,lower_quantile_idx]
    
    return lower_quantile_arr, upper_quantile_arr

def compute_coverage_QGPT(interval_center, Y_predict, output_list, past_window, target_quantiles):
    
    CI_length = len(output_list)
    
    interval_center_arr = np.array(interval_center)[-CI_length:]
    Y_predict_arr = np.array(Y_predict)[-CI_length:]
    
    quantile_iter = int(len(target_quantiles)/2)
    
    for i in range(quantile_iter):
        
        upper_quantile = target_quantiles[i]
        lower_quantile = target_quantiles[-(i+1)]
        
        lower_quantile_arr, upper_quantile_arr = compute_CI_QGPT(output_list, past_window, i, -(i+1))
    
        upper_CI = interval_center_arr + upper_quantile_arr
        lower_CI = interval_center_arr + lower_quantile_arr
    
        coverage_tf = (Y_predict_arr <= upper_CI) * (Y_predict_arr >= lower_CI)
        coverage_ratio = np.sum(coverage_tf) / len(coverage_tf)
    
        CI_width_arr = upper_quantile_arr-lower_quantile_arr
        print("quantile pair: {} , {}".format(upper_quantile, lower_quantile))
        print("coverage_ratio: {}".format(coverage_ratio))
        print("average CI width: {}".format(np.mean(CI_width_arr)))
        print("-------------------------------")

def plot_QGPT_results(saving_dir, interval_center, Y_predict, output_list, past_window, target_quantiles, ylim):
    
    CI_length = len(output_list)
    
    interval_center_arr = np.array(interval_center)[-CI_length:]
    Y_predict_arr = np.array(Y_predict)[-CI_length:]
    
    quantile_iter = int(len(target_quantiles)/2)
    
    for i in range(quantile_iter):
        
        upper_quantile = target_quantiles[i]
        lower_quantile = target_quantiles[-(i+1)]
        
        lower_quantile_arr, upper_quantile_arr = compute_CI_QGPT(output_list, past_window, i, -(i+1))
    
        upper_CI = interval_center_arr + upper_quantile_arr
        lower_CI = interval_center_arr + lower_quantile_arr
    
        coverage_tf = (Y_predict_arr <= upper_CI) * (Y_predict_arr >= lower_CI)
        coverage_ratio = np.sum(coverage_tf) / len(coverage_tf)
    
        CI_width_arr = upper_quantile_arr-lower_quantile_arr

        saving = saving_dir + "SPCI-T_{}_{}.pdf".format(upper_quantile, lower_quantile)
        print("quantile pair: {} , {}".format(upper_quantile, lower_quantile))
        print("coverage_ratio: {}".format(coverage_ratio))
        print("average CI width: {}".format(np.mean(CI_width_arr)))
        print("-------------------------------")
        plt.figure(figsize=(10, 5))
        plt.ylim(ylim)
        plt.plot(Y_predict_arr, label=r'$Y$')
        plt.fill_between(np.arange(len(Y_predict_arr)), upper_CI, lower_CI, alpha=0.3, label=r'$\hat{C}(X,\alpha)$')
        plt.xlabel('time index', fontsize=15)
        plt.legend(bbox_to_anchor=(1, -0.05), fancybox=True, shadow=True, ncol=2, fontsize=12)
        plt.title("SPCI-T", fontsize=20)
        plt.savefig(saving, bbox_inches = 'tight',pad_inches = 0.0)

def compute_quantile_loss(outputs: torch.Tensor, targets: torch.Tensor, desired_quantiles: torch.Tensor) -> torch.Tensor:
    """
    This function compute the quantile loss a.k.a. pinball loss separately per each sample, time-step, and quantile.

    Parameters
    ----------
    outputs: torch.Tensor
        The outputs of the model (num_prediction_step * batch_size * num_quantiles)
    targets: torch.Tensor
        The observed target for each horizon (num_prediction_step * batch_size)
    desired_quantiles: torch.Tensor
        A tensor representing the desired quantiles, of shape (num_quantiles)

    Returns
    -------
    losses_array: torch.Tensor
        a tensor [num_samples x num_horizons x num_quantiles] containing the quantile loss for each sample,time-step and
        quantile.
    """

    # compute the actual error between the observed target and each predicted quantile
    # TODO: consider mask in resid_y 
    # errors = targets.reshape((targets.size()[0]*targets.size()[1],1)) - outputs.reshape((outputs.size()[0]*outputs.size()[1],outputs.size()[2])) 
    errors = targets.unsqueeze(-1) - outputs # (num_samples * num_time_steps * num_quantiles)

    # compute the loss separately for each sample,time-step,quantile
    losses_array = torch.max((desired_quantiles - 1) * errors, desired_quantiles * errors) # element-wise max

    # sum losses over quantiles and average across time and observations: scalar
    return (losses_array.sum(dim=-1)).mean(dim=-1).mean()
