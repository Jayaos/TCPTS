import torch
from torch import nn, Tensor
import math
from torch.utils.data import Dataset


class DatasetQB(Dataset):

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
    
def collate_fn_QB(batch):

    resid_x_batch, feature_x_batch, resid_y_batch, featureresid_y_batch, feature_y_batch = zip(*batch)

    resid_x_tensor = torch.Tensor(resid_x_batch) # batch_size * past_window
    feature_x_tensor = torch.Tensor(feature_x_batch) # batch_size * past_window * feature_dim 
    resid_y_tensor = torch.Tensor(resid_y_batch) # batch_size * num_step_prediction
    #featureresid_y_tensor = torch.Tensor(featureresid_y_batch) # not required for QuantileBERT
    #feature_y_tensor = torch.Tensor(feature_y_batch) # not required for QuantileBERT

    batch_size, past_window, feature_dim = feature_x_tensor.size()

    resid_x_tensor = resid_x_tensor.reshape((batch_size, past_window, 1)) # batch_size * past_window * 1
    x_tensor = torch.cat([resid_x_tensor, feature_x_tensor], axis=-1) # batch_size * past_window * (feature_dim+1)

    return x_tensor, resid_y_tensor


class QuantileBERT(nn.Module):

    """
    stacked Transformer Encoder for quantile regression
    Input: sequential features for endoder padded by max_seq_len and sequential features decoder
        sequential features for encoder: (batch_size * past_window * feature_dim)
        sequential features for decoder: (batch_size * step_prediction * feature_dim) 
        we dont need sequential features for decoder if the aim is single step prediction
    Output: predicted values for pre-defined quantiles (batch_size * step_prediction * # of pre-defined quantiles)
    """
    
    def __init__(self, dim_feature: int, dim_model: int, num_head: int, dim_ff: int, num_layer: int, 
                 target_quantiles: list, dropout: float = 0.1, batch_first: bool=True):
        super(QuantileBERT, self).__init__()

        Encoder_Layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_head, dim_feedforward=dim_ff, 
                                                   dropout=dropout, batch_first=batch_first)
        self.Encoder = nn.TransformerEncoder(Encoder_Layer, num_layers=num_layer)

        self.encoder_input_linear = nn.Linear(dim_feature, dim_model) # this will work as embedding layer for features
        self.output_linear = nn.Linear(dim_model, len(target_quantiles)) # no activation


    def forward(self, src: Tensor, src_mask: Tensor, src_key_padding_mask: Tensor):

        src_emb = self.encoder_input_linear(src)
        src_emb = src_emb * math.sqrt(self.dim_model)
        src_emb = self.positional_encoding(src_emb)

        outputs = self.Encoder(src_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        return self.output_linear(outputs)
    
