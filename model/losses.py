"""
    @author: Jay Lago, NIWC Pacific, 55280
"""
import torch
import torch.nn as nn

class ResMQLoss(nn.Module):
    def __init__(self, hyp):
        super(ResMQLoss, self).__init__()
        self.quantiles = hyp['quantiles']
        self.num_tgt_features = hyp['num_tgt_features']
        self.device = hyp['device']
        self.register_parameter('w_lin', nn.Parameter(torch.tensor(0.0)))
        self.register_parameter('w_res', nn.Parameter(torch.tensor(0.0)))

    def forward(self, y, y_preds, y_quants):
        '''
            lin_err: [batch, time, features]
            res_err: [batch, time, features, quantiles]
        '''
        lin_err = self.get_lin_err(y, y_preds)
        res_err = self.get_res_err(lin_err, y_quants)
        mse_lin = torch.mean(torch.mean(torch.square(lin_err), dim=1))
        mse_res = torch.mean(torch.mean(torch.square(res_err), dim=1))
        return torch.exp(-self.w_lin) * mse_lin + torch.exp(-self.w_res) * mse_res + (self.w_lin + self.w_res)
    
    def get_lin_err(self, y, y_preds):
        return y - y_preds

    def get_res_err(self, residuals, y_quants):
        res_err = torch.zeros_like(y_quants, dtype=y_quants.dtype).to(y_quants.device)
        for ii in range(self.num_tgt_features):
            for jj, q in enumerate(self.quantiles):
                err = residuals[..., ii] - y_quants[..., ii, jj]
                res_err[..., ii, jj] = torch.max(q*err, (q-1)*err)
        return res_err
