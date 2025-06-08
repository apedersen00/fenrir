import torch
from tqdm import tqdm
from snntorch import functional as SF
import os

def train_epoch(config, model, trainloader, criterion, optimizer, device, scheduler=None, current_epoch=0, grad_log_path=None):
    model.train()
    epoch_loss_accum = []
    epoch_lr_accum = []
    
    pbar_desc = f"Epoch {current_epoch}/{config['num_epochs']-1} Training"
    pbar = tqdm(trainloader, desc=pbar_desc, leave=False)

    for batch_idx, (data, labels) in enumerate(pbar):
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        spk_rec = model(data) # Model forward pass
        loss = criterion(spk_rec, labels)
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step() # CosineAnnealingLR steps per batch/iteration

        # --- Metrics ---
        current_loss_val = loss.item()
        batch_accuracy_percent = 0.0
        with torch.no_grad():
            batch_accuracy = SF.accuracy_rate(spk_rec, labels)
            if isinstance(batch_accuracy, torch.Tensor): # Should return float
                batch_accuracy = batch_accuracy.item()
            batch_accuracy_percent = batch_accuracy * 100

        # --- Gradient Norm ---
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        # --- Logging & Progress Bar ---
        postfix_dict = {
            "loss": f"{current_loss_val:.4f}",
            "acc": f"{batch_accuracy_percent:.2f}%",
            "grad": f"{total_grad_norm:.2e}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
        }
        pbar.set_postfix(postfix_dict)
        
        loss_to_accumulate = current_loss_val / config.get("n_timesteps", 1.0) # Avoid division by zero if key missing
        epoch_loss_accum.append(loss_to_accumulate)
        epoch_lr_accum.append(optimizer.param_groups[0]["lr"])

        # --- Gradient Logging (first batch only) ---
        if batch_idx == 0 and grad_log_path:
            try:
                with open(grad_log_path, 'a') as f_log:
                    f_log.write(f"\nEpoch {current_epoch}, Batch {batch_idx} - Gradient Norms (Detailed):\n")
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm_val = param.grad.norm().item()
                            f_log.write(f"  {name:40}: {grad_norm_val:.6f}\n")
                        else:
                            f_log.write(f"  {name:40}: No gradient\n")
                    f_log.write("-" * 60 + "\n")
            except IOError as e:
                print(f"Warning: Could not write to gradient log file {grad_log_path}. Error: {e}")
                
    avg_epoch_loss = sum(epoch_loss_accum) / len(epoch_loss_accum) if epoch_loss_accum else 0
    return epoch_loss_accum, epoch_lr_accum, avg_epoch_loss


def test_model(config, model, testloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    pbar = tqdm(testloader, desc="Testing", leave=False)
    with torch.no_grad():
        for data, labels in pbar:
            images, labels = data.to(device), labels.to(device)
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0) # Accumulate weighted loss

            accuracy_val = SF.accuracy_rate(outputs, labels) # Returns fraction
            correct_predictions += accuracy_val * labels.size(0)
            total_samples += labels.size(0)
            
            pbar.set_postfix({
                "avg_loss": f"{total_loss / total_samples:.4f}",
                "acc": f"{(correct_predictions / total_samples) * 100:.2f}%"
            })

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    overall_accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
    return avg_loss, overall_accuracy