'''
    @author: Jay Lago, NIWC Pacific, 55280
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
            self,
            layers,
            norm_layer=None
    ):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross):
        # x: [N, L, C] --> [N, L, C]
        sattns = []
        xattns = []
        new_x = torch.clone(x)
        for layer in self.layers:
            new_x, sattn, xattn = layer(new_x, cross)
            sattns.append(sattn)
            xattns.append(xattn)
        x = x + self.norm(new_x)
        return x, sattns, xattns


class DecoderLayer(nn.Module):
    def __init__(
            self,
            self_attention,
            cross_attention,
            d_model,
            d_ff,
            dropout=0.1,
            activation='gelu'
    ):
        super(DecoderLayer, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x, cross):
        # x: [N, L, C] --> [N, L, C]
        new_x, sattn = self.self_attention(x, x, x)
        x = x + self.norm1(self.dropout(new_x))
        new_x, xattn = self.cross_attention(x, cross, cross)
        y = x = x + self.norm2(self.dropout(new_x))
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = x + self.norm3(x + y)
        return x, sattn, xattn

