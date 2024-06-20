import torch
from typing import Tuple
import numpy as np
from torch import nn, Tensor
import math
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view


def build_strided_residue_QuantileTransformer(residue_sequence, window_len, num_prediction_step):
    
    strided_residue_sequences = sliding_window_view(residue_sequence, window_len+num_prediction_step)
    strided_residue_input_sequences = strided_residue_sequences[:,:-num_prediction_step]
    strided_residue_prediction_input_sequences = strided_residue_sequences[:,-(num_prediction_step+1):-1]
    strided_residue_prediction_sequences = strided_residue_sequences[:,-num_prediction_step:]
    
    return strided_residue_input_sequences, strided_residue_prediction_input_sequences, strided_residue_prediction_sequences

def build_strided_feature_QuantileTransformer(feature_sequence, window_len, num_prediction_step):
    
    strided_feature_sequences = sliding_window_view(feature_sequence, window_len+num_prediction_step, axis=0)
    strided_feature_input_sequences = strided_feature_sequences[:,:,:-num_prediction_step]
    strided_feature_prediction_input_sequences = strided_feature_sequences[:,:,-(num_prediction_step+1):-1]
    strided_feature_input_sequences = np.transpose(strided_feature_input_sequences, (0,2,1))
    strided_feature_prediction_input_sequences = np.transpose(strided_feature_prediction_input_sequences, (0,2,1))
    
    return strided_feature_input_sequences, strided_feature_prediction_input_sequences

class DatasetQuantileTransformer(Dataset):

    # build datset for QuantileTransformer implementation
    
    def __init__(self, residX, featureX, residY, featureresidY, featureY):
        self.residX = residX
        self.featureX = featureX
        self.residY = residY
        self.featureresidY = featureresidY
        self.featureY = featureY

    def __len__(self):
        return len(self.residY)

    def __getitem__(self,idx):
        return self.residX[idx,:], self.featureX[idx,:], self.residY[idx], self.featureresidY[idx], self.featureY[idx]

def collate_fn_QuantileTransformer(batch):

    resid_x_batch, feature_x_batch, resid_y_batch, featureresid_y_batch, feature_y_batch = zip(*batch)

    resid_x_tensor = torch.Tensor(resid_x_batch) # batch_size * past_window
    feature_x_tensor = torch.Tensor(feature_x_batch) # batch_size * past_window * feature_dim 
    resid_y_tensor = torch.Tensor(resid_y_batch) # batch_size * num_step_prediction
    featureresid_y_tensor = torch.Tensor(featureresid_y_batch) # batch_size * num_step_prediction
    feature_y_tensor = torch.Tensor(feature_y_batch) # batch_size * num_step_prediction * feature_dim 

    batch_size, num_step_prediction, feature_dim = feature_y_tensor.size()
    _, past_window = resid_x_tensor.size()

    resid_x_tensor = resid_x_tensor.reshape((batch_size, past_window, 1)) # batch_size * past_window * 1
    x_tensor = torch.cat([resid_x_tensor, feature_x_tensor], axis=-1) # batch_size * past_window * (feature_dim+1)

    featureresid_y_tensor = featureresid_y_tensor.reshape((batch_size, num_step_prediction, 1))
    y_tensor = torch.cat([featureresid_y_tensor, feature_y_tensor], axis=-1) # input for decoder, batch_size * num_step_prediction * (feature_dim+1)

    return x_tensor, y_tensor, resid_y_tensor

class QuantileTransformer(nn.Module):

    """
    Transformer for multi-step quantile regression
    Input: sequential features for endoder padded by max_seq_len and sequential features decoder
        sequential features for encoder: (batch_size * max_seq_len * feature_dim)
        sequential features for decoder: (batch_size * step_prediction-1 * feature_dim) 
        we dont need sequential features for decoder if the aim is single step prediction
    Output: predicted values for pre-defined quantiles (batch_size * step_prediction * # of pre-defined quantiles)
    """

    def __init__(self, dim_feature: int, dim_model: int, num_head: int, dim_ff: int, 
                 num_encoder_layers: int, num_decoder_layers: int, target_quantiles: list, dropout: float=0.1, batch_first: bool=True):
        super(QuantileTransformer, self).__init__()
        # we do not need embedding layer since we assume the features are all continuous real
        self.dim_model = dim_model
        self.target_quantiles = target_quantiles
        self.positional_encoding = PositionalEncoding(dim_model, dropout)
        self.transformer = nn.Transformer(d_model=dim_model,
                                       nhead=num_head,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_ff,
                                       dropout=dropout,
                                       batch_first=batch_first)
        # linear projection of the features of dim_feature into dim_model
        # use the same embedding for endoer and decoder
        self.input_linear = nn.Linear(dim_feature, dim_model)
        # linear projection of the representation of dim_model into len(quantiles) 
        self.output_linear = nn.Linear(dim_model, len(target_quantiles)) # no activation

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_key_padding_mask: Tensor,
                tgt_key_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        
        src_emb = self.input_linear(src)
        src_emb = src_emb * math.sqrt(self.dim_model)
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.input_linear(tgt)
        tgt_emb = tgt_emb * math.sqrt(self.dim_model)
        tgt_emb = self.positional_encoding(tgt_emb)

        outputs = self.transformer(src_emb, tgt_emb, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=None, 
                                   src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, 
                                   memory_key_padding_mask=memory_key_padding_mask)
        
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
