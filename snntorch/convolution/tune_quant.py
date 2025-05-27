import time
import torch
from torch.utils.data import DataLoader
import tonic
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
    reset_cache=False
)

train_loader = DataLoader(
    disk_cached_trainset,
    batch_size=64,
    shuffle=True,
    collate_fn=tonic.collation.PadTensors(),
    num_workers=6,
)
# --- Load the trained model ---
model = ThreeConvPoolingNet().to(device)
model.load_state_dict(torch.load('best_model.pt'))  # Adjust path if needed
print("Loaded previously trained model")

# --- Create a new optimizer with reduced learning rate ---
new_lr = 0.0001  # 5-10x reduction from previous rate
optimizer = torch.optim.Adam(model.parameters(), lr=new_lr)
print(f"Created new optimizer with learning rate: {new_lr}")

criterion = torch.nn.CrossEntropyLoss()

# --- Fine-tuning for 1 epoch ---
model.train()
start_time = time.time()
scalar_total_loss = 0.0
total_samples = 0
correct = 0

for batch_idx, (frames, targets) in enumerate(train_loader):
    frames = frames.to(device)
    targets = targets.to(device)

    optimizer.zero_grad()
    
    # Handle both cases (with or without spike return)
    try:
        outputs, (spikes1, spikes2, spikes3) = model(frames, return_spikes=True)
        # Apply regularization if your model returns spikes
        firing_rates = [spikes1.mean(), spikes2.mean(), spikes3.mean()]
        target_rate = 0.1
        firing_rate_penalty = sum([(rate - target_rate)**2 for rate in firing_rates])
        loss = criterion(outputs, targets) + 0.001 * firing_rate_penalty
    except:
        # If model doesn't support returning spikes
        outputs = model(frames)
        loss = criterion(outputs, targets)
    
    loss.backward()
    
    # Add gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
    
    optimizer.step()
    
    scalar_total_loss += loss.item() * frames.size(0)
    total_samples += frames.size(0)
    preds = outputs.argmax(dim=1)
    correct += (preds == targets).sum().item()
    
    if (batch_idx + 1) % 10 == 0:
        print(f"Fine-tuning | Batch {batch_idx+1}/{len(train_loader)} | "
              f"Loss: {loss.item():.4f} | "
              f"Acc: {correct/total_samples:.4f}")

epoch_time = time.time() - start_time
final_acc = correct / total_samples
final_loss = scalar_total_loss / total_samples

print(f"Fine-tuning completed in {epoch_time:.2f}s | "
      f"Final Loss: {final_loss:.4f} | Final Acc: {final_acc:.4f}")

# Save the fine-tuned model
torch.save(model.state_dict(), 'model_finetuned.pt')
print("Fine-tuned model saved.")

# Compare with previous best
print(f"Previous accuracy: ~0.90 | New accuracy: {final_acc:.4f}")
improvement = (final_acc - 0.90) * 100
print(f"Improvement: {improvement:.2f}%")