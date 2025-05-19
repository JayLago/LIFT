'''
    @author: Jay Lago, NIWC Pacific, 55280
'''
from typing import Dict
import torch
import torch.nn as nn

from .embedding import DataEmbedding
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
from .attention import AttentionLayer, FullAttention


#------------------------------------------------------------------------------
# Transformer for multi-quantile forecast of residual of linear model
#------------------------------------------------------------------------------
class LIFT(nn.Module):
    def __init__(self, hyp: Dict):
        super(LIFT, self).__init__()
        self.dim_model = hyp['dim_model']
        self.dim_feedforward = hyp['dim_feedforward']
        self.num_heads = hyp['num_heads']
        self.num_enc_layers = hyp['num_enc_layers']
        self.num_dec_layers = hyp['num_dec_layers']
        self.dropout = hyp['dropout']
        self.enc_seq_len = hyp['enc_seq_len']
        self.dec_seq_len = hyp['dec_seq_len']
        self.num_enc_features = hyp['num_enc_features']
        self.num_dec_features = hyp['num_dec_features']
        self.num_tgt_features = hyp['num_tgt_features']
        self.activation = hyp['activation']
        self.quantiles = hyp['quantiles']
        self.dim_out = len(hyp['quantiles'])
        self.emb_kernel_size = hyp['emb_kernel_size']

        # Normalization
        self.enc_norm_layer = nn.LayerNorm(self.num_enc_features)
        self.dec_norm_layer = nn.LayerNorm(self.num_dec_features)
        
        # Linear autoregressive submodels
        self.linear_predictors = nn.ModuleList([
            nn.Linear(self.enc_seq_len, self.dec_seq_len) for _ in range(self.num_tgt_features)
        ])

        # Embedding
        self.embed_enc_features = DataEmbedding(
            chan_in=self.num_enc_features,
            dim_model=self.dim_model,
            enc_len=self.enc_seq_len,
            dec_len=self.dec_seq_len,
            kernel_size=self.emb_kernel_size
        )
        self.embed_dec_features = DataEmbedding(
            chan_in=self.num_dec_features,
            dim_model=self.dim_model,
            enc_len=self.enc_seq_len,
            dec_len=self.dec_seq_len,
            kernel_size=self.emb_kernel_size
        )
        
        # Encoder
        self.encoder = Encoder(
            attn_layers = [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            attention_dropout=self.dropout,
                            output_attention=True
                        ),
                        d_model=self.dim_model,
                        n_heads=self.num_heads,
                    ),
                    d_model=self.dim_model,
                    d_ff=self.dim_feedforward,
                    dropout=self.dropout,
                    activation=self.activation,
                ) for _ in range(self.num_enc_layers)
            ],
            norm_layer=nn.LayerNorm(self.dim_model),
        )
        
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    # Self attention
                    AttentionLayer(
                        FullAttention(
                            attention_dropout=self.dropout,
                            output_attention=False
                        ),
                        d_model=self.dim_model,
                        n_heads=self.num_heads,
                    ),
                    # Cross attention with encoder
                    AttentionLayer(
                        FullAttention(
                            attention_dropout=self.dropout,
                            output_attention=True
                        ),
                        d_model=self.dim_model,
                        n_heads=self.num_heads,
                    ),
                    # Feedforward
                    d_model=self.dim_model,
                    d_ff=self.dim_feedforward,
                    dropout=self.dropout,
                    activation=self.activation,
                ) for _ in range(self.num_dec_layers)
            ],
            norm_layer=nn.LayerNorm(self.dim_model),
        )

        # Prediction layers
        self.quantile_predictors = nn.ModuleList([
            nn.Linear(self.dim_model, self.dim_out) for _ in range(self.num_tgt_features)
        ])
    
    def forward(self, x_enc, x_dec):
        """
        Args:
            x_enc (torch.Tensor): [batch, time_in, channels]
            x_dec (torch.Tensor): [batch, time_out, channels]
        Returns:
            tuple: A tuple containing the following elements:
            - preds (torch.Tensor): Predictions from the linear prediction model. Shape: [batch, time_out, channels]
            - quantiles (torch.Tensor): Quantiles for each point in the forecast. Shape: [batch, time_out, channels, quantiles]
            - enc_attns (torch.Tensor): Attention weights for the encoder. Shape: [batch, num_heads, time_in, time_out]
            - self_attns (torch.Tensor): Attention weights for the self-attention mechanism 
                in the decoder. Shape: [batch, num_heads, time_in, time_out]
            - cross_attns (torch.Tensor): Attention weights for the cross-attention mechanism 
                in the decoder. Shape: [batch, num_heads, time_in, time_out]
        """
        
        # Linear prediction model
        preds = torch.zeros([x_dec.shape[0], x_dec.shape[1], self.num_tgt_features], dtype=x_dec.dtype).to(x_dec.device)
        for ii, linear_predictor in enumerate(self.linear_predictors):
            preds[:, :, ii:ii+1] = linear_predictor(x_enc[..., ii:ii+1].permute(0, 2, 1)).permute(0, 2, 1)

        # Normalize
        x_enc = self.enc_norm_layer(x_enc)
        x_dec = self.dec_norm_layer(x_dec)

        # Encode
        x_enc = self.embed_enc_features(x_enc)
        x_enc, enc_attns = self.encoder(x_enc)

        # Decode
        x_dec = self.embed_dec_features(x_dec)
        x_dec, self_attns, cross_attns = self.decoder(x_dec, x_enc)
        
        # Map latent features to quantiles for each point in the forecast
        quantiles = torch.zeros([x_dec.shape[0], x_dec.shape[1], self.num_tgt_features, self.dim_out], dtype=x_dec.dtype).to(x_dec.device)
        for ii, quantile_predictor in enumerate(self.quantile_predictors):
            quantiles[:, :, ii, :] = quantile_predictor(x_dec)

        return preds, quantiles, enc_attns, self_attns, cross_attns
