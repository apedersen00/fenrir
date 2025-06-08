import torch
import torch.optim as optim
import yaml
import argparse
import os
import time
import numpy as np
from snntorch import functional as SF
from src.model import *
from src.data_utils import get_dataloaders
from src.trainer import train_epoch, test_model
from src.utils import plot_loss_lr, setup_gradient_log_file, get_device

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(args):
    # --- Load Configuration ---
    config = load_config(args.config_file)
    model_name = config['model']
    
    # Create directories if they don't exist
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['data_dir'], exist_ok=True)

    # --- Setup ---
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = get_device(config.get('device', 'auto'))
    print(f"Using device: {device}")

    grad_log_full_path = None
    if config.get('gradient_log_file'):
        grad_log_full_path = setup_gradient_log_file(config['log_dir'], config['gradient_log_file'])

    # --- Data ---
    print("Loading data...")
    trainloader, testloader = get_dataloaders(config)
    print(f"Trainloader size: {len(trainloader.dataset)}, Testloader size: {len(testloader.dataset)}")

    # --- Model ---
    print(f"Initializing model: '{model_name}'...")
    match model_name:
        case 'FenrirNet':
            model = FenrirNet(config).to(device) 
        case 'FenrirFC':
            model = FenrirFC(config).to(device)
        case _:
            print(f"Unknown model '{model_name}'.")

    # --- Optimizer, Scheduler, Criterion ---
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["lr"],
        betas=tuple(config["betas"]) # Ensure betas is a tuple
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["t_0"],
        eta_min=config["eta_min"]
    )
    criterion = SF.mse_count_loss( # Use your actual SF here
        correct_rate=config["correct_rate"],
        incorrect_rate=config["incorrect_rate"]
    )

    # --- Load Model (if specified) ---
    start_epoch = 0
    if args.load_model_path:
        if os.path.exists(args.load_model_path):
            print(f"Loading model from {args.load_model_path}")
            checkpoint = torch.load(args.load_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                 start_epoch = checkpoint['epoch'] + 1
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print(f"Warning: Model path {args.load_model_path} not found. Training from scratch.")
    else:
        print("Training from scratch.")

    # --- Training Loop ---
    all_losses = []
    all_lrs = []
    best_test_accuracy = 0.0

    print(f"======= Starting Training for {config['num_epochs'] - start_epoch} epochs =======")
    training_start_time = time.time()

    for epoch in range(start_epoch, config['num_epochs']):
        epoch_start_time = time.time()
        
        batch_losses, batch_lrs, avg_train_loss = train_epoch(
            config, model, trainloader, criterion, optimizer, device, 
            scheduler, epoch, grad_log_full_path
        )
        all_losses.extend(batch_losses)
        all_lrs.extend(batch_lrs)
        
        epoch_duration = time.time() - epoch_start_time
        
        # --- Test ---
        test_loss, test_accuracy = test_model(config, model, testloader, criterion, device)
        
        print(f"Epoch: {epoch}/{config['num_epochs']-1} | "
              f"Avg Train Loss: {avg_train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.2f}% | "
              f"Epoch Time: {epoch_duration:.2f}s")

        # --- Save Model (best and/or periodically) ---
        if args.save_model_name:
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                best_model_name = f"{args.save_model_name}_{model_name}_best.pth"
                best_model_path = os.path.join(config['model_dir'], best_model_name)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'loss': avg_train_loss, # or test_loss
                    'test_accuracy': test_accuracy
                }, best_model_path)
                print(f"New best model saved to {best_model_path} (Accuracy: {test_accuracy:.2f}%)")

    total_training_time = time.time() - training_start_time
    print(f"======= Training Finished in {total_training_time:.2f}s =======")
    print(f"Best Test Accuracy: {best_test_accuracy:.2f}%")

    # --- Plotting ---
    if args.plot:
        plot_save_dir = config['log_dir']
        plot_loss_lr(all_losses, all_lrs, save_path=plot_save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN Gesture Training Script")
    parser.add_argument('--config_file', type=str, default='config.yaml', help='Path to the configuration YAML file.')
    
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to load a pre-trained model state_dict from.')
    parser.add_argument('--save_model_name', type=str, default="fenrir_dvsgesture", help='Base name for saving the trained model. _best.pth" will be appended.')
    
    parser.add_argument('--plots', action='store_true', help='Save plot of loss and LR to the log directory after training.')

    cli_args = parser.parse_args()
    main(cli_args)