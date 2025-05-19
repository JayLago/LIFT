"""
    @author: Jay Lago, NIWC Pacific, 55280
"""
import numpy as np
import json
from tqdm import tqdm

import torch
import torchinfo
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from .data_module import TransformerDataModule
from .train_utils import validation_step, validation_diagnostic_plots, test_diagnostic_plots, EarlyStopping

def training(config, model, train_dataloader, val_dataloader, test_dataloader,
             loss_fn, optimizer, scheduler, early_stopping):
    
    # Save configuration to file
    with open(f"./results/{config.model_name}/train_config.json", 'w') as f:
        f.write(json.dumps(config))

    # Print model summary
    print(f"Using device: {config.device}")
    print(torchinfo.summary(
        model,
        [(config.enc_seq_len, config.num_enc_features), 
         (config.dec_seq_len, config.num_dec_features)],
        batch_dim=0,
        col_names=('input_size', 'output_size', 'num_params', 'mult_adds'),
        verbose=0
    ))
    model.to(config.device)

    # Initialize optimizer, learning rate scheduler, and early stopping
    optimizer = optim.Adam(model.parameters(), lr=hyp['learning_rate'])
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.9,
        patience=2,
        threshold=1e-4,
        threshold_mode='rel',
    )
    early_stopping = EarlyStopping(patience=7, min_delta=1e-3)


    # Initialize model data loaders
    data_module = TransformerDataModule(config)
    data_module.prepare_data()
    train_dataloader = data_module.setup(stage='fit')
    val_dataloader = data_module.setup(stage='validate')
    test_dataloader = data_module.setup(stage='test')
    rand_val_idx = np.random.randint(0, config.batch_size, size=3)
    
    train_history = []
    val_history = []
    epoch_history = []
    for epoch in range(config.num_epochs):
        with tqdm(train_dataloader, desc=f"Epoch {epoch}/{config.num_epochs}", unit='batch') as pbar:
            for batch in pbar:
                # Tran step
                model.train()
                enc_inputs = batch[0].to(config.device)
                dec_inputs = batch[1].to(config.device)
                tgt_outputs = batch[2].to(config.device)
                if enc_inputs.isnan().any() or dec_inputs.isnan().any() or tgt_outputs.isnan().any():
                    print("[ERROR] NaN values detected in input batch.")
                    break
                
                outputs = model(enc_inputs, dec_inputs)[0]
                loss = loss_fn(tgt_outputs, outputs)

                # Update step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_history.append(loss.item())
                
                # Progress bar
                pbar.set_postfix({'Loss': f"{np.mean(epoch_history[-config.batch_size:]):1.5f}"})
                pbar.update()
        
        # Save train loss
        train_history.append(np.mean(epoch_history[-config.batch_size:]))
        
        # Validation step
        val_history, _ = validation_step(model, loss_fn, config, val_dataloader, val_history)
        
        # Early stopping
        early_stopping(val_history[-1])
        if early_stopping.early_stop:
            # Finish up here
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_history[-1],
            }, f"./results/{config.model_name}/checkpoint_{epoch}.tar")
            validation_diagnostic_plots(
                model, config, loss_fn, epoch, val_dataloader, 
                data_module, rand_val_idx, train_history, val_history, 
                with_attn=True
            )
            torch.save(model.state_dict(), f"./results/{config.model_name}/model_epoch{epoch}.pth")
            test_diagnostic_plots(model, config, loss_fn, test_dataloader, data_module)
            return
        
        # Adjust learning rate
        scheduler.step(val_history[-1])
        print(f"Validation loss: {val_history[-1]}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        # Checkpoint and diagnostic plot
        if epoch % config.val_plot_freq==0 or epoch==0 or epoch==config.num_epochs-1:
            # Checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_history[-1],
            }, f"./results/{config.model_name}/checkpoint_{epoch}.tar")
            # Diagnostic plots
            validation_diagnostic_plots(
                model, config, loss_fn, epoch, val_dataloader, 
                data_module, rand_val_idx, train_history, val_history, 
                with_attn=True
            )
    
    # Save model
    torch.save(model.state_dict(), f"./results/{config.model_name}/model_epoch{epoch}.pth")
    
    # Final test diagnostic
    test_diagnostic_plots(model, config, loss_fn, test_dataloader, data_module)
