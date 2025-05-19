'''
    @author: Jay Lago, NIWC Pacific, 55280
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
            self,
            attn_layers,
            norm_layer
        ):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        # x: [N, L, C] --> [N, L, C]
        attns = []
        new_x = torch.clone(x)
        for attn_layer in self.attn_layers:
            new_x, attn = attn_layer(new_x)
            attns.append(attn)
        x = x + self.norm(new_x)
        return x, attns


class EncoderLayer(nn.Module):
    def __init__(
            self,
            attention,
            d_model,
            d_ff,
            dropout=0.1,
            activation='gelu'
        ):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x):
        # x: [N, L, C] --> [N, L, C]
        new_x, attn = self.attention(x, x, x)
        y = x = x + self.norm1(self.dropout(new_x))
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        x = x + self.norm2(x + y)
        return x, attn


