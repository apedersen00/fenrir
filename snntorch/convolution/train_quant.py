from f_quant_net import FQuantNetInt, ThreeConvPoolingNet
import tonic.datasets as datasets
import tonic.transforms as transforms
from tonic import DiskCachedDataset
import torch
from torch.utils.data import DataLoader
import tonic, time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CropTo32(object):
    def __call__(self, frames):
        return frames[..., 1:33, 1:33]

class OnlyPositive(object):
    def __call__(self, frames):
        return frames[:, 1:2, :, :]

frame_transform = transforms.Compose([
    transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size, time_window=1000),
    CropTo32(),
    OnlyPositive(),
])

trainset = datasets.NMNIST(
    save_to='./data/nmnist',
    train=True,
    transform=frame_transform,
)

frames, target = trainset[0]
print("Sample nonzeros:", (frames != 0).sum())
print("Sample max:", frames.max())
print("Sample min:", frames.min())

disk_cached_trainset = DiskCachedDataset(
    trainset,
    cache_path='./data/nmnist_cache',
    reset_cache=True
)

train_loader = DataLoader(
    disk_cached_trainset,
    batch_size=64,
    shuffle=True,
    collate_fn=tonic.collation.PadTensors(),
    num_workers=6,
)

# --- Model, Loss, Optimizer ---
model = ThreeConvPoolingNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Replace StepLR with ReduceLROnPlateau for adaptive scheduling
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

# Number of epochs to train
num_epochs = 10

# Early stopping parameters
patience = 5
best_loss = float('inf')
patience_counter = 0

# Training history
history = {
    'train_loss': [],
    'train_acc': [],
    'epoch_times': []
}

# --- Multi-epoch Training Loop ---
for epoch in range(num_epochs):
    start_time = time.time()
    
    model.train()
    scalar_total_loss = 0.0
    total_samples = 0
    correct = 0
    
    for batch_idx, (frames, targets) in enumerate(train_loader):
        frames = frames.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs, (spikes1, spikes2, spikes3) = model(frames, return_spikes=True)
        
        # Base loss
        loss = criterion(outputs, targets)
        
        # Add firing rate regularization for more stable neurons
        firing_rates = [spikes1.mean(), spikes2.mean(), spikes3.mean()]
        target_rate = 0.1  # Target sparsity - lower means fewer spikes
        firing_rate_penalty = sum([(rate - target_rate)**2 for rate in firing_rates])
        loss = loss + 0.001 * firing_rate_penalty
        
        loss.backward()
        
        # Add gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        scalar_total_loss += loss.item() * frames.size(0)
        total_samples += frames.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        
        if batch_idx == 0 and epoch == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"DEBUG: {name} grad norm = {param.grad.norm().item():.6f}")
                else:
                    print(f"DEBUG: {name} grad is None")
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {correct/total_samples:.4f}")
    
    # End of epoch statistics
    epoch_loss = scalar_total_loss / total_samples
    epoch_acc = correct / total_samples
    epoch_time = time.time() - start_time
    
    history['train_loss'].append(epoch_loss)
    history['train_acc'].append(epoch_acc)
    history['epoch_times'].append(epoch_time)
    
    print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s | "
          f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
    
    # Update learning rate based on validation loss
    scheduler.step(epoch_loss)
    # After scheduler.step(epoch_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current learning rate: {current_lr:.6f}")
    # Early stopping check
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        patience_counter = 0
        # Save the best model
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"Model improved! Saved checkpoint at epoch {epoch+1}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# Load the best model
model.load_state_dict(torch.load('best_model.pt'))
print("Training completed! Best model restored.")

# Plot training curves
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'])
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('training_curves.png')
