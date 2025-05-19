'''
    ----------------------------------------------------------------------
    Distribution statement A. Approved for public release. Distribution is 
    unlimited. This work was supported by the Office of Naval Research and
    the Naval Information Warfare Center Pacific.
    ----------------------------------------------------------------------
    @author: Jay Lago, NIWC Pacific, 55280
'''
# Basics
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import datetime as dt
import time
import argparse
import json

# PyTorch
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# Library
from model.models import LIFT
from model.losses import ResMQLoss
from utils.train_utils import model_training_loop, EarlyStopping
from utils.utils import get_cuda_summary

assert torch.__version__.startswith("2")
torch.set_float32_matmul_precision('high')

DAY = 96
DEVICE = get_cuda_summary()
SEED = 1997
np.random.seed(SEED)
torch.manual_seed(SEED)


# Main program
def main(args):
    base_dir = "."
    data_path = f"{base_dir}/data/dataset_{args.min_cont_days}day.pkl"
    train_list = f"{base_dir}/data/station_train_list.csv"
    test_list = f"{base_dir}/data/station_test_list.csv"
    sim_time = time.strftime('%Y%m%d_%H%M')
    model_name = (f"LIFT_{sim_time}")
    if not os.path.exists(f"./results/{model_name}"):
        os.makedirs(f"./results/{model_name}")

    # Standard hyperparameters
    hyp = {}
    hyp['model_name'] = model_name
    hyp['data_path'] = data_path
    hyp['train_list'] = train_list
    hyp['test_list'] = test_list
    hyp['val_split'] = args.val_split
    hyp['accelerator'] = 'gpu'
    hyp['device'] = DEVICE
    hyp['num_sub'] = -1
    hyp['num_loader_workers'] = args.num_loader_workers
    hyp['tqdm'] = args.tqdm

    # Main user hyperparameters
    hyp['num_epochs'] = args.num_epochs
    hyp['val_plot_freq'] = args.val_plot_freq
    hyp['learning_rate'] = args.learning_rate
    hyp['enc_seq_len'] = args.enc_seq_len
    hyp['dec_seq_len'] = args.dec_seq_len
    hyp['tgt_seq_len'] = args.tgt_seq_len
    hyp['batch_size'] = args.batch_size
    hyp['dim_model'] = args.dim_model
    hyp['dim_feedforward'] = args.dim_feedforward
    hyp['num_heads'] = args.num_heads
    hyp['num_enc_layers'] = args.num_enc_layers
    hyp['num_dec_layers'] = args.num_dec_layers
    hyp['emb_kernel_size'] = args.emb_kernel_size
    hyp['activation'] = args.activation
    hyp['dropout'] = args.dropout
    hyp['loss_fn'] = 'ResMQLoss'
    hyp['quantiles'] = [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]

    # Data configs
    hyp['num_days'] = args.min_cont_days
    hyp['val_split'] = args.val_split
    hyp['window_size'] = hyp['enc_seq_len'] + hyp['tgt_seq_len']
    hyp['window_step'] = args.window_step
    hyp['indices_path'] = (f"./data/indices/indices_{hyp['num_days']}days_"
                           f"{hyp['window_size']}win_{hyp['window_step']}step.pkl")
    hyp['new_indices'] = not os.path.exists(hyp['indices_path'])

    # Input/output config
    hyp['data_cols'] = ['foF2', 'hmF2', 'TEC',
                        'sza', 'IRI_F2_fo', 'IRI_F2_hm', 'IRI_F2_B_bot', 'IRI_F2_B_top',
                        'sYOS', 'cYOS', 'sDOY', 'cDOY', 'sHOD', 'cHOD', 'sMOH', 'cMOH',
                        'F107', 'DST', 'Kp', 'SSN']
    hyp['tgt_cols_idx'] = [0, 1, 2]
    hyp['past_cols_idx'] = [i for i in range(len(hyp['data_cols']))]
    hyp['future_cols_idx'] = [i for i in range(3, len(hyp['data_cols'])-4)]
    hyp['forecast_units'] = ['MHz', 'km', 'TECU']
    hyp['num_tgt_features'] = len(hyp['tgt_cols_idx'])
    hyp['num_enc_features'] = len(hyp['past_cols_idx'])
    hyp['num_dec_features'] = max(len(hyp['future_cols_idx']), 1)

    # Save hyperparameters
    with open(f"./results/{hyp['model_name']}/0_hyperparams.json", 'w') as f:
        f.write(json.dumps(hyp))
    
    # Create the model
    model = LIFT(hyp)

    # Set loss function
    print(f"Using loss function: {hyp['loss_fn']}")
    loss_fn = ResMQLoss(hyp)
    
    # Initialize optimizer, learning rate scheduler, and early stopping
    optimizer = optim.Adam(model.parameters(), lr=hyp['learning_rate'])
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.9,
        patience=3,
        threshold=1e-3,
        threshold_mode='rel',
    )
    early_stopping = EarlyStopping(patience=8, min_delta=1e-4)

    # Train it
    model_training_loop(hyp, model, optimizer, loss_fn, early_stopping, scheduler)
    

# Command line program
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LIFT')
    parser.add_argument('--min_cont_days',      type=int, default=4)
    parser.add_argument('--window_step',        type=int, default=12)
    parser.add_argument('--num_epochs',         type=int, default=1000)
    parser.add_argument('--val_plot_freq',      type=int, default=100)
    parser.add_argument('--learning_rate',      type=float, default=1e-4)
    parser.add_argument('--batch_size',         type=int, default=256)
    parser.add_argument('--val_split',          type=float, default=0.8)
    parser.add_argument('--enc_seq_len',        type=int, default=3*DAY)
    parser.add_argument('--dec_seq_len',        type=int, default=1*DAY)
    parser.add_argument('--tgt_seq_len',        type=int, default=1*DAY)
    parser.add_argument('--dim_model',          type=int, default=128)
    parser.add_argument('--dim_feedforward',    type=int, default=128)
    parser.add_argument('--num_heads',          type=int, default=8)
    parser.add_argument('--num_enc_layers',     type=int, default=1)
    parser.add_argument('--num_dec_layers',     type=int, default=1)
    parser.add_argument('--emb_kernel_size',    type=int, default=4)
    parser.add_argument('--attention_type',     type=str, default='full')
    parser.add_argument('--activation',         type=str, default='gelu')
    parser.add_argument('--dropout',            type=float, default=0.1)
    parser.add_argument('--num_loader_workers', type=int, default=0)
    parser.add_argument('--k_fold',             type=int, default=-1)
    parser.add_argument('--tqdm',               type=bool, default=True)
    args = parser.parse_args()

    train_start = dt.datetime.now()
    main(args)
    train_stop = dt.datetime.now() - train_start
    print(f'\nTotal elapsed time: {train_stop}')

