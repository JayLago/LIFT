'''
    @author: Jay Lago, NIWC Pacific, 55280
'''
import torch
import torch.nn as nn
import math

#------------------------------------------------------------------------------
# Embedding Layers
#------------------------------------------------------------------------------
class DataEmbedding(nn.Module):
    def __init__(
            self,
            chan_in,
            dim_model,
            enc_len,
            dec_len,
            dropout=0.1,
            kernel_size=4,
    ):
        super(DataEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = PositionalEncoder(
            d_model=dim_model,
            enc_len=enc_len,
            dec_len=dec_len
        )
        self.value_embedding = ValueEmbedding(
            chan_in=chan_in,
            d_model=dim_model,
            kernel_size=kernel_size,
        )

    def forward(self, x, enc_flag=False):
        x = self.position_embedding(x, enc_flag) + self.value_embedding(x)
        return self.dropout(x)


class ValueEmbedding(nn.Module):
    def __init__(
            self,
            chan_in,
            d_model,
            kernel_size
        ):
        super(ValueEmbedding, self).__init__()
        self.conv_layer = nn.Conv1d(
            in_channels=chan_in,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            padding_mode='reflect'
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class PositionalEncoder(nn.Module):
    def __init__(
            self,
            d_model,
            enc_len,
            dec_len,
            max_len=5000
        ):
        super(PositionalEncoder, self).__init__()
        self.enc_len = enc_len
        self.dec_len = dec_len
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, enc_flag=False):
        return self.pe[:, :x.size(1)]


