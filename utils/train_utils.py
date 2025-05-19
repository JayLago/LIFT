"""
    @author: Jay Lago, NIWC Pacific, 55280
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
mpl.rcParams['font.size'] = 20
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['font.family'] = 'Sans-Serif'
mpl.rcParams['text.usetex'] = False
import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torchinfo

from .data_module import TransformerDataModule
from .utils import toNumpyCPU, get_quantile_index

figtype = 'png'
DPI = 150
SEED = 1997
np.random.seed(SEED)
torch.manual_seed(SEED)


class EarlyStopping:
    def __init__(self, patience=6, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def save_checkpoint(epoch, model, optimizer, history, filename):
    torch.save({'epoch': epoch, 
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': history[-1]},
        filename
    )


def load_checkpoint(model, filename):
    model.load_state_dict(torch.load(filename))


##############################################################################
# Training loop
##############################################################################
def model_training_loop(hyp, model, optimizer, loss_fn, early_stopping, scheduler):

    # Get train/val/test loaders
    data_module = TransformerDataModule(hyp)
    data_module.prepare_data()
    test_dataloader = data_module.setup(stage='test')
    train_dataloader = data_module.setup(stage='fit')
    val_dataloader = data_module.setup(stage='validate')
    
    # Report model summary and send to the GPU
    print(f"Using device: {hyp['device']}")
    print(torchinfo.summary(
        model,
        [(hyp['enc_seq_len'], hyp['num_enc_features']), 
         (hyp['dec_seq_len'], hyp['num_dec_features'])],
        batch_dim=0,
        col_names=('input_size', 'output_size', 'num_params', 'mult_adds'),
        verbose=0
    ))
    model.to(hyp['device'])

    # Set up random indices for validation diagnostic plots
    rand_val_idx = np.random.randint(0, hyp['batch_size'], size=3)

    # Training outputs
    train_history = []
    val_history = []
    epoch_history = []
    for epoch in range(hyp['num_epochs']):
        with tqdm(train_dataloader, desc=f"Epoch {epoch}/{hyp['num_epochs']}", unit='batch') as pbar:
            for batch in pbar:
                # Tran step
                model.train()
                enc_inputs = batch[0].to(hyp['device'])
                dec_inputs = batch[1].to(hyp['device'])
                tgt_outputs = batch[2].to(hyp['device'])
                preds, quants = model(enc_inputs, dec_inputs)[:2]
                loss = loss_fn(tgt_outputs, preds, quants)

                # Update step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_history.append(loss.item())
                
                # Progress bar
                pbar.set_postfix({'Loss': f"{np.mean(epoch_history[-hyp['batch_size']:]):1.5f}"})
                pbar.update()
        
        # Save train loss
        train_history.append(np.mean(epoch_history[-hyp['batch_size']:]))
        
        # Validation step
        val_history, _ = validation_step(model, loss_fn, hyp, val_dataloader, val_history)
        
        # Early stopping
        early_stopping(val_history[-1])
        if early_stopping.early_stop:
            # Finish up here
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_history[-1],
            }, f"./results/{hyp['model_name']}/checkpoint_{epoch}.tar")
            validation_diagnostic_plots(
                model, hyp, loss_fn, epoch, val_dataloader, 
                data_module, rand_val_idx, train_history, val_history, 
            )
            torch.save(model.state_dict(), f"./results/{hyp['model_name']}/model_epoch{epoch}.pth")
            test_diagnostic_plots(model, hyp, loss_fn, test_dataloader, data_module)
            return
        
        # Adjust learning rate
        scheduler.step(val_history[-1])
        print(f"Validation loss: {val_history[-1]}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        # Checkpoint and diagnostic plot
        if epoch % hyp['val_plot_freq']==0 or epoch==0 or epoch==hyp['num_epochs']-1:
            # Checkpoint
            save_checkpoint(
                epoch, model, optimizer, train_history,
                f"./results/{hyp['model_name']}/checkpoint_{epoch}.tar"
            )
            # Diagnostic plots
            validation_diagnostic_plots(
                model, hyp, loss_fn, epoch, val_dataloader, 
                data_module, rand_val_idx, train_history, val_history, 
            )
    
    # Save final model
    torch.save(model.state_dict(), f"./results/{hyp['model_name']}/model_epoch{epoch}.pth")
    
    # Generate final test diagnostic plots
    test_diagnostic_plots(model, hyp, loss_fn, test_dataloader, data_module)



##############################################################################
# Validation loops
##############################################################################
def validation_step(model, loss_fn, hyp, val_dataloader, val_history):
    model.eval()
    with torch.no_grad():
        mean_val_loss = []
        for val_batch in tqdm(val_dataloader, desc="Validation"):
            enc_inputs = val_batch[0].to(hyp['device'])
            dec_inputs = val_batch[1].to(hyp['device'])
            tgt_outputs = val_batch[2].to(hyp['device'])
            preds, dist = model(enc_inputs, dec_inputs)[:2]
            val_loss = loss_fn(tgt_outputs, preds, dist)
            mean_val_loss.append(val_loss.item())
        mean_val_loss = np.mean(mean_val_loss)
        val_history.append(mean_val_loss)
    return val_history, mean_val_loss



##############################################################################
# Validation set diagnostics
##############################################################################
def validation_diagnostic_plots(model, hyp, loss_fn, epoch, val_dataloader, data_module,
                                rand_val_idx, train_history, val_history):

    _, val_df, _ = data_module.get_dataframes()
    val_indices = data_module.val_indices[0]
    ridx = rand_val_idx[0]
    num_tgts = hyp['num_tgt_features']
    tgt_cols = hyp['data_cols'][:hyp['num_tgt_features']]

    CI_median = get_quantile_index(model.quantiles, q=0.5)

    # Run a single batch for plotting
    model.eval()
    with torch.no_grad():
        val_batch = next(iter(val_dataloader))
        enc_inputs = val_batch[0].to(hyp['device'])
        dec_inputs = val_batch[1].to(hyp['device'])
        tgt_outputs = val_batch[2].to(hyp['device'])
        preds, dist = model(enc_inputs, dec_inputs)[:2]
    
    enc_inputs = toNumpyCPU(enc_inputs)
    dec_inputs = toNumpyCPU(dec_inputs)
    tgt_outputs = toNumpyCPU(tgt_outputs)
    preds = toNumpyCPU(preds)
    dist = toNumpyCPU(dist)
    
    fig = plt.figure(figsize=(42, 18), constrained_layout=True)
    outer = gridspec.GridSpec(1, num_tgts+1, wspace=0.2, hspace=0.2)
    for ii in range(num_tgts):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[ii], wspace=0.1, hspace=0.1)
        past_x_axis = np.arange(hyp['enc_seq_len'])
        future_x_axis = np.arange(hyp['enc_seq_len'], hyp['enc_seq_len'] + hyp['tgt_seq_len'])
        # Data
        ax = plt.Subplot(fig, inner[0])
        ax.plot(past_x_axis, enc_inputs[ridx, :, ii], 'b-o', label="Past values")
        ax.plot(future_x_axis, tgt_outputs[ridx, :, ii], 'k-x', label="True future values")
        # Forecast distributions
        for iq in range(len(hyp['quantiles'])//2):
            iq_upper = len(hyp['quantiles']) - 1 - iq
            ax.fill_between(future_x_axis,
                            preds[ridx, :, ii] + dist[ridx, :, ii, iq],
                            preds[ridx, :, ii] + dist[ridx, :, ii, iq_upper],
                            color='blue', alpha=hyp['quantiles'][iq_upper])
        ax.plot(future_x_axis, preds[ridx, :, ii] + dist[ridx, :, ii, CI_median], 'r-', label='Median pred.')
        this_rmse = np.sqrt(np.mean((tgt_outputs[ridx, :, ii] - (preds[ridx, :, ii] + dist[ridx, :, ii, CI_median]))**2))
        
        ax.axvline(x=future_x_axis[0], color='k', label='past/future')
        ax.legend(loc='upper left', fontsize=12)
        station_id = val_df.iloc[val_indices[ridx][0]]['station_id']
        date_start = val_df.iloc[val_indices[ridx][0]]['timestamp']
        date_stop = val_df.iloc[val_indices[ridx][1]]['timestamp']

        ax.set_title(f"Station: {station_id} | RMSE: {this_rmse:.4f}")
        ax.set_ylabel(f'{tgt_cols[ii]}')
        ax.set_xticklabels([])
        ax.grid(alpha=0.75)
        fig.add_subplot(ax)
        col_names = hyp['data_cols']
        # Past covariates
        ax = plt.Subplot(fig, inner[1])
        for jj in range(len(hyp['past_cols_idx'])):
            if col_names[hyp['past_cols_idx'][jj]] not in data_module.time_feature_cols:
                norm_cov = enc_inputs[ridx, :, jj]
                norm_cov = (norm_cov - norm_cov.mean()) / (1e-6 + norm_cov.std())
                ax.plot(past_x_axis, norm_cov, '-', label=f"{col_names[hyp['past_cols_idx'][jj]]}")
        # Future covariates
        if len(hyp['future_cols_idx'])>0:
            for jj in range(len(hyp['future_cols_idx'])):
                if col_names[hyp['future_cols_idx'][jj]] not in data_module.time_feature_cols:
                    norm_cov = dec_inputs[ridx, :, jj]
                    norm_cov = (norm_cov - norm_cov.mean()) / (1e-6 + norm_cov.std())
                    ax.plot(future_x_axis, norm_cov, '--', label=f"{col_names[hyp['future_cols_idx'][jj]]}")
        # Past/future separator
        ax.axvline(x=future_x_axis[0], color='k', label='past/future')
        ax.legend(loc='upper left', fontsize=12)
        ax.set_ylabel('Normalized covariates')
        ax.set_xlabel('Time (days)')
        if (hyp['enc_seq_len']+hyp['tgt_seq_len']/96)>4:
            tick_freq = '24h'
        else:
            tick_freq = '12h'
        date_tick_list = [ts.strftime('%b%d %H%M') for ts in 
                          pd.date_range(start=date_start, end=date_stop, freq=tick_freq, inclusive='left').to_list()]
        ax.set_xticks(np.linspace(0, future_x_axis[-1], len(date_tick_list)))
        ax.set_xticklabels(date_tick_list, rotation=0)
        # ax.tick_params(labelrotation=22.5)
        ax.grid(alpha=0.75)
        fig.add_subplot(ax)
    
    # plt.suptitle(f"Samples From Validation Set")
    fname = f"./results/{hyp['model_name']}/epoch_{epoch}_validation.{figtype}"
    plt.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close()
    
    # Plot loss history
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    plt.plot(train_history, 'r-o', lw=1, label='train')
    plt.plot(val_history, 'b-o', lw=1, label='validation')
    plt.legend(loc='upper right', fontsize=16)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.title(f"Training / Validation History")
    fname = f"./results/{hyp['model_name']}/epoch_{epoch}_loss.{figtype}"
    plt.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close()


##############################################################################
# Test set diagnostics
##############################################################################
def test_diagnostic_plots(model, hyp, loss_fn, test_dataloader, data_module):

    tgt_cols = hyp['data_cols'][:hyp['num_tgt_features']]
    rand_val_idx = np.random.randint(0, hyp['batch_size'], size=9)
    _, _, test_df = data_module.get_dataframes()
    test_indices = data_module.test_indices
    
    CI_outer_low = get_quantile_index(model.quantiles, q=0.1)
    CI_outer_high = get_quantile_index(model.quantiles, q=0.9)
    CI_inner_low = get_quantile_index(model.quantiles, q=0.25)
    CI_inner_high = get_quantile_index(model.quantiles, q=0.75)
    CI_median = get_quantile_index(model.quantiles, q=0.5)

    # Just one batch for the diagnostic plot
    model.eval()
    with torch.no_grad():
        seg = next(iter(test_dataloader))
        enc_inputs = seg[0].to(hyp['device'])
        dec_inputs = seg[1].to(hyp['device'])
        tgt_outputs = seg[2].to(hyp['device'])
        preds, dist = model(enc_inputs, dec_inputs)[:2]
    
    enc_inputs = toNumpyCPU(enc_inputs)
    dec_inputs = toNumpyCPU(dec_inputs)
    tgt_outputs = toNumpyCPU(tgt_outputs)
    preds = toNumpyCPU(preds)
    dist = toNumpyCPU(dist)

    for jj in range(hyp['num_tgt_features']):
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(40, 30), constrained_layout=True)
        ax = ax.flatten()
        for ii, ridx in enumerate(rand_val_idx):
            axs = ax[ii]
            past_x_axis = np.arange(hyp['enc_seq_len'])
            future_x_axis = np.arange(hyp['enc_seq_len'], hyp['enc_seq_len'] + hyp['tgt_seq_len'])
            
            l1 = axs.plot(past_x_axis, enc_inputs[ridx, :, jj], 'b-o', label="Past values")
            l2 = axs.plot(future_x_axis, tgt_outputs[ridx, :, jj], 'k-x', label="True future values")

            axs.fill_between(future_x_axis,
                            preds[ridx, :, jj] + dist[ridx, :, jj, CI_outer_low],
                            preds[ridx, :, jj] + dist[ridx, :, jj, CI_outer_high],
                            color='blue', alpha=0.25)
            axs.fill_between(future_x_axis,
                            preds[ridx, :, jj] + dist[ridx, :, jj, CI_inner_low],
                            preds[ridx, :, jj] + dist[ridx, :, jj, CI_inner_high],
                            alpha=0.5, color='green')
            l3 = axs.plot(future_x_axis, preds[ridx, :, jj] + dist[ridx, :, jj, CI_median], 'r-', label='Median pred.')
            this_rmse = np.sqrt(np.mean((tgt_outputs[ridx, :, jj] - (preds[ridx, :, jj] - dist[ridx, :, jj, CI_median]))**2))

            l4 = axs.axvline(x=future_x_axis[0], color='k', label='past/future')
            
            blue_patch = mpatches.Patch(color='blue', alpha=0.25, label='90% CI')
            green_patch = mpatches.Patch(color='green', alpha=0.5, label='IQR')
            axs.legend(handles=[l1[0], l2[0], l3[0], l4, blue_patch, green_patch], loc='upper left', fontsize=12)

            station_id = test_df.iloc[test_indices[ridx][0]]['station_id']
            date_start = test_df.iloc[test_indices[ridx][0]]['timestamp']
            date_stop = test_df.iloc[test_indices[ridx][1]]['timestamp']
            axs.set_title(f"Station: {station_id} | RMSE: {this_rmse:.4f}")
            axs.set_ylabel(f'{tgt_cols[jj]}')
            axs.set_xlabel('Time (days)')
            if (hyp['enc_seq_len']+hyp['tgt_seq_len']/96)>4:
                tick_freq = '24h'
            else:
                tick_freq = '12h'
            date_tick_list = [ts.strftime('%b%d %H%M') for ts in 
                            pd.date_range(start=date_start, end=date_stop, freq=tick_freq, inclusive='left').to_list()]
            axs.set_xticks(np.linspace(0, future_x_axis[-1], len(date_tick_list)))
            axs.set_xticklabels(date_tick_list, rotation=0)
            axs.grid(alpha=0.75)

        # plt.show()
        fname = f"./results/{hyp['model_name']}/epoch_{hyp['num_epochs']}_test_{tgt_cols[jj]}.{figtype}"
        plt.savefig(fname, dpi=DPI, bbox_inches="tight")
        plt.close()

    #
    # Compute error across full test data set
    #
    test_dataloader = data_module.setup(stage='test')
    test_dataset = data_module._get_individual_test_segment_dataset()
    num_segments = len(test_dataloader)*hyp['batch_size']
    all_test_seg = np.zeros(shape=(hyp['num_tgt_features'], num_segments, hyp['tgt_seq_len']))
    all_pred_seg = np.zeros(shape=(hyp['num_tgt_features'], num_segments, hyp['tgt_seq_len']))
    quant_p10 = np.zeros(shape=(hyp['num_tgt_features'], num_segments, hyp['tgt_seq_len']))
    quant_p50 = np.zeros(shape=(hyp['num_tgt_features'], num_segments, hyp['tgt_seq_len']))
    quant_p90 = np.zeros(shape=(hyp['num_tgt_features'], num_segments, hyp['tgt_seq_len']))
    with torch.no_grad():
        iseg = 0
        for batch in tqdm(test_dataloader, desc="Final testing"):
            enc_inputs = batch[0].to(hyp['device'])
            dec_inputs = batch[1].to(hyp['device'])
            tgt_outputs = batch[2].to(hyp['device'])

            preds, quants = model(enc_inputs, dec_inputs)[:2]
            
            if hyp['num_tgt_features']>1:
                y_true = toNumpyCPU(tgt_outputs).squeeze()
                preds = toNumpyCPU(preds).squeeze()
                quants = toNumpyCPU(quants).squeeze()
            else:
                y_true = toNumpyCPU(tgt_outputs)
                preds = toNumpyCPU(preds)
                quants = toNumpyCPU(quants)
            
            yhat_10 = preds + quants[..., CI_outer_low]
            yhat_50 = preds + quants[..., CI_median]
            yhat_90 = preds + quants[..., CI_outer_high]

            for ib in range(hyp['batch_size']):
                for ii in range(hyp['num_tgt_features']):
                    all_test_seg[ii, iseg, :] = y_true[ib, :, ii]
                    all_pred_seg[ii, iseg, :] = yhat_50[ib, :, ii]
                    quant_p10[ii, iseg, :] = y_true[ib, :, ii]<yhat_10[ib, :, ii]
                    quant_p50[ii, iseg, :] = y_true[ib, :, ii]<yhat_50[ib, :, ii]
                    quant_p90[ii, iseg, :] = y_true[ib, :, ii]<yhat_90[ib, :, ii]
                iseg += 1
    
    # Error stats
    all_residuals = all_test_seg - all_pred_seg
    ssr = np.zeros(shape=(hyp['num_tgt_features'], hyp['tgt_seq_len']))
    sst = np.zeros(shape=(hyp['num_tgt_features'], hyp['tgt_seq_len']))
    r2 = np.zeros(shape=(hyp['num_tgt_features'], hyp['tgt_seq_len']))
    rmse = np.zeros(shape=(hyp['num_tgt_features'], num_segments))
    mae = np.zeros(shape=(hyp['num_tgt_features'], num_segments))
    mape = np.zeros(shape=(hyp['num_tgt_features'], num_segments))
    lower_10th = np.zeros(shape=(hyp['num_tgt_features'], num_segments))
    upper_10th = np.zeros(shape=(hyp['num_tgt_features'], num_segments))
    for ii in range(hyp['num_tgt_features']):
        rmse[ii, :] = np.sqrt(np.mean(np.square(all_residuals[ii, :, :]), axis=-1))
        mae[ii, :] = np.mean(np.abs(all_residuals[ii, :, :]), axis=-1)
        mape[ii, :] = 100*np.mean(np.abs(all_residuals[ii, :, :] / (all_test_seg[ii, :, :]+1e-6)), axis=-1)
        lower_10th[ii, :] = np.percentile(a=np.abs(all_residuals[ii, :, :]), q=10, axis=-1)
        upper_10th[ii, :] = np.percentile(a=np.abs(all_residuals[ii, :, :]), q=90, axis=-1)
        ssr[ii, :] = np.sum(np.square(all_residuals[ii, :, :]), axis=0)
        sst[ii, :] = np.sum(np.square(all_test_seg[ii, :, :] - np.mean(all_test_seg[ii, :, :], axis=0)), axis=0)
        r2[ii, :] = 1.0 - (ssr[ii, :] / sst[ii, :])

    # Plots
    for ii in range(hyp['num_tgt_features']):
        print(f'---> Plotting test diagnostics for {tgt_cols[ii]}')
        # Plot histogram of median forecast error
        print('Plotting histogram of median prediction errors...')
        fig = plt.figure(figsize=(20, 10), constrained_layout=True)
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], figure=fig)
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1], sharex=ax0)
        ax0.hist(all_residuals[ii, :, :].flatten(), bins=100, color='blue', alpha=0.5)
        ax0.axvline(x=np.mean(all_residuals[ii, :, :]), color='k', label=r'Mean')
        # ax.set_xticks(list(np.arange(-10, 10, 2)))
        ax0.legend(loc='upper right', fontsize=24)
        # ax0.set_xlabel(r"$x-\hat{x}$ (MHz)")
        ax0.set_ylabel(r'Count')
        ax0.grid(alpha=0.5)
        plt.suptitle(f"{hyp['forecast_units'][ii]} - RMSE: {rmse[ii].mean():2.3f}, MAE: {mae[ii].mean():2.3f}, MAPE: {mape[ii].mean():2.3f} / "
                     f"Mean Abs. 10th {np.mean(lower_10th[ii,:]):2.3f} / 90th {np.mean(upper_10th[ii, :]):2.3f}")
        ax1.boxplot(all_residuals[ii, :, :].flatten(), vert=False, showfliers=False)
        ax1.grid(alpha=0.5)
        ax1.set_xlabel(f"Median test errors ({hyp['forecast_units'][ii]})")
        # plt.show()
        fname = f"./results/{hyp['model_name']}/epoch_{hyp['num_epochs']}_test_metrics_median_{tgt_cols[ii]}.{figtype}"
        plt.savefig(fname, dpi=DPI, bbox_inches="tight")
        plt.close()

        # Plot histogram of forecast statistics
        print('Plotting histogram of prediction stats...')
        fig = plt.figure(figsize=(20, 10), constrained_layout=True)
        gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1], figure=fig)
        metrics = [(rmse, "RMSE"), (mae, "MAE"), (mape, "MAPE")]
        for jj, (metric, label) in enumerate(metrics):
            ax1 = fig.add_subplot(gs[1, jj])
            ax0 = fig.add_subplot(gs[0, jj], sharex=ax1)
            ax1.boxplot(metric[ii, :], vert=False, showfliers=False)
            ax1.set_xlabel(f"{label}")
            ax1.grid(alpha=0.5)
            ax0.hist(metric[ii, :], bins=100, color='blue', alpha=0.5)
            ax0.axvline(x=metric[ii].mean(), color='k', lw=3, label=r'Mean')
            ax0.legend(loc='upper right', fontsize=24)
            ax0.set_ylabel(f'Count')
            ax0.set_title(f'Avg. {label}: {metric[ii].mean():2.3f}')
            ax0.grid(alpha=0.5)
        plt.suptitle(f"{tgt_cols[ii]}")
        fname = f"./results/{hyp['model_name']}/epoch_{hyp['num_epochs']}_test_metrics_mse-mae-mape_{tgt_cols[ii]}.{figtype}"
        plt.savefig(fname, dpi=DPI, bbox_inches="tight")
        plt.close()

        # Plot histogram of  percentile errors
        print('Plotting histogram of percentile errors...')
        fig, ax = plt.subplots(1, 3, figsize=(20, 10), constrained_layout=True)
        ax = ax.flatten()
        ax[0].hist(np.mean(quant_p10[ii, :, :], axis=-1), bins=30, color='blue', alpha=0.5)
        ax[0].axvline(x=np.mean(quant_p10[ii, :, :]), color='k', lw=3, label=r'Mean')
        ax[0].legend(loc='upper right', fontsize=24)
        ax[0].set_xticks(np.arange(0, 1, 0.2))
        ax[0].set_xlabel(r"$x < \hat{F}^{-1}(0.1)$")
        ax[0].set_ylabel(r'Count')
        ax[0].set_title(f'10th: {np.mean(quant_p10[ii, :, :]):2.3f}')
        ax[0].grid(alpha=0.5)
        ax[1].hist(np.mean(quant_p50[ii, :, :], axis=-1), bins=30, color='blue', alpha=0.5)
        ax[1].axvline(x=np.mean(quant_p50[ii, :, :]), color='k', lw=3, label=r'Mean')
        ax[1].legend(loc='upper right', fontsize=24)
        ax[1].set_xticks(np.arange(0, 1, 0.2))
        ax[1].set_xlabel(r"$x < \hat{F}^{-1}(0.5)$")
        ax[1].set_ylabel(r'Count')
        ax[1].set_title(f'50th: {np.mean(quant_p50[ii, :, :]):2.3f}')
        ax[1].grid(alpha=0.5)
        ax[2].hist(np.mean(quant_p90[ii, :, :], axis=-1), bins=30, color='blue', alpha=0.5)
        ax[2].axvline(x=np.mean(quant_p90[ii, :, :]), color='k', lw=3, label=r'Mean')
        ax[2].legend(loc='upper left', fontsize=24)
        ax[2].set_xticks(np.arange(0, 1, 0.2))
        ax[2].set_xlabel(r"$x < \hat{F}^{-1}(0.9)$")
        ax[2].set_ylabel(r'Count')
        ax[2].set_title(f'90th: {np.mean(quant_p90[ii, :, :]):2.3f}')
        ax[2].grid(alpha=0.5)
        fname = f"./results/{hyp['model_name']}/epoch_{hyp['num_epochs']}_test_metrics_10-90_{tgt_cols[ii]}.{figtype}"
        plt.savefig(fname, dpi=DPI, bbox_inches="tight")
        plt.close()

        # Worst performing segments
        print('Plotting worst performing segments...')
        fig, ax = plt.subplots(5, 5, figsize=(80, 60), constrained_layout=True)
        ax = ax.flatten()
        worst_mses = np.sort(rmse[ii, :])[-25:]
        worst_indices = np.argsort(rmse[ii, :])[-25:]
        for jj, iworst in enumerate(worst_indices):
            bad_seg = test_dataset.__getitem__(iworst)
            enc_inputs = bad_seg[0].to(hyp['device']).unsqueeze(0)
            dec_inputs = bad_seg[1].to(hyp['device']).unsqueeze(0)
            tgt_outputs = bad_seg[2].to(hyp['device']).unsqueeze(0)
            preds, dist = model(enc_inputs, dec_inputs)[:2]
            
            if hyp['num_tgt_features']>1:
                preds = toNumpyCPU(preds).squeeze()
                dist = toNumpyCPU(dist).squeeze()
                enc_inputs = toNumpyCPU(enc_inputs).squeeze()
                dec_inputs = toNumpyCPU(dec_inputs).squeeze()
                tgt_outputs = toNumpyCPU(tgt_outputs).squeeze()
            else:
                preds = toNumpyCPU(preds.squeeze(0))
                dist = toNumpyCPU(dist.squeeze(0))
                enc_inputs = toNumpyCPU(enc_inputs.squeeze(0))
                dec_inputs = toNumpyCPU(dec_inputs.squeeze(0))
                tgt_outputs = toNumpyCPU(tgt_outputs.squeeze(0))
            
            preds = preds + dist[..., CI_median]

            axs = ax[jj]
            past_x_axis = np.arange(hyp['enc_seq_len'])
            future_x_axis = np.arange(hyp['enc_seq_len'], hyp['enc_seq_len'] + hyp['tgt_seq_len'])
            l1 = axs.plot(past_x_axis, enc_inputs[:, ii], 'b-o', label=r"Past observed")
            l2 = axs.plot(future_x_axis, tgt_outputs[:, ii], 'k-x', label=r"Future observed")

            axs.fill_between(future_x_axis,
                                preds[:, ii] + dist[:, ii, CI_outer_low],
                                preds[:, ii] + dist[:, ii, CI_outer_high],
                                color='blue', alpha=0.25)
            axs.fill_between(future_x_axis,
                                preds[:, ii] + dist[:, ii, CI_inner_low],
                                preds[:, ii] + dist[:, ii, CI_inner_high],
                                alpha=0.5, color='green')
            l3 = axs.plot(future_x_axis, preds[:, ii] + dist[:, ii, CI_median], 'r-', label='Median pred.')

            l4 = axs.axvline(x=future_x_axis[0], color='k', linestyle='--', label=r'Forecast start')
            blue_patch = mpatches.Patch(color='blue', alpha=0.25, label=r'Forecast min-max')
            green_patch = mpatches.Patch(color='green', alpha=0.5, label=r'Forecast $10^{th}-90^{th}$ perc.')

            station_id = test_df.iloc[test_indices[iworst][0]]['station_id']
            date_start = test_df.iloc[test_indices[iworst][0]]['timestamp']
            date_stop = test_df.iloc[test_indices[iworst][1]]['timestamp']
            axs.set_title(f"Station: {station_id} | MSE: {worst_mses[jj]:.4f}")
            axs.set_ylabel(f'{tgt_cols[ii]}')
            # axs.set_xlabel(r'Time')
            if (hyp['enc_seq_len']+hyp['tgt_seq_len']/96)>4:
                tick_freq = '24h'
            else:
                tick_freq = '12h'
            date_tick_list = [ts.strftime('%b%d %H%M') for ts in 
                                pd.date_range(start=date_start, end=date_stop, freq=tick_freq, inclusive='left').to_list()]
            axs.set_xticks(np.linspace(0, future_x_axis[-1], len(date_tick_list)))
            axs.set_xticklabels(date_tick_list)
            axs.grid(alpha=0.75)

        # plt.show()
        fname = f"./results/{hyp['model_name']}/epoch_{hyp['num_epochs']}_test_worst_segments_{tgt_cols[ii]}.{figtype}"
        plt.savefig(fname, dpi=DPI, bbox_inches="tight")
        plt.close()
