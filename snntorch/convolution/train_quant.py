from f_quant_net import FQuantNetInt
import tonic.datasets as datasets
import tonic.transforms as transforms
from tonic import DiskCachedDataset
import torch
from torch.utils.data import DataLoader
import tonic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CropTo32(object):
    def __call__(self, frames):
        return frames[..., 1:33, 1:33]

class OnlyPositive(object):
    def __call__(self, frames):
        return frames[:, 1:2, :, :]

frame_transform = transforms.Compose([
    transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size, n_time_bins=200),
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
)

train_loader = DataLoader(
    disk_cached_trainset,
    batch_size=32,
    shuffle=True,
    collate_fn=tonic.collation.PadTensors(),
    num_workers=4,
)

# --- Model, Loss, Optimizer ---
model = FQuantNetInt().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

quant_reg_strength = 0.01  # Try 0.001, tune if needed!

# --- Training Loop (1 epoch) ---
model.train()
scalar_total_loss = 0.0
total_samples = 0
correct = 0


for batch_idx, (frames, targets) in enumerate(train_loader):
    frames = frames.to(device)
    targets = targets.to(device)

    optimizer.zero_grad()
    outputs = model(frames)

    loss = criterion(outputs, targets)

    # Quantization loss
    quant_loss = 0.0
    quant_loss += model.conv1.quant_reg_loss
    quant_loss += model.conv2.quant_reg_loss
    # (Add other QuantConv2d layers if you have them)

    loss_total = loss + quant_reg_strength * quant_loss
    loss_total.backward()

    # Print some gradient norms for debug (first batch)
    if batch_idx == 0:
        print(frames[0][0])
        

    optimizer.step()

    scalar_total_loss += loss.item() * frames.size(0)
    total_samples += frames.size(0)
    preds = outputs.argmax(dim=1)
    correct += (preds == targets).sum().item()

    if batch_idx == 0:
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"DEBUG: {name} grad norm = {param.grad.norm().item():.6f}")
            else:
                print(f"DEBUG: {name} grad is None")

    if (batch_idx + 1) % 10 == 0:
        print(f"Batch {batch_idx+1}/{len(train_loader)} | "
              f"Loss: {loss.item():.4f} | "
              f"Acc: {correct/total_samples:.4f}")
        
    if (batch_idx + 1) % 100 == 0:
        print(f"\n--- Batch {batch_idx + 1} ---")
        for name, conv in [('conv1', model.conv1), ('conv2', model.conv2)]:
            w = conv.conv.weight.detach().cpu()
            print(f"{name} weights: min {w.min():.2f}, max {w.max():.2f}, mean {w.mean():.2f}")
print(f"Finished 1 epoch. Avg Loss: {scalar_total_loss/total_samples:.4f} | "
      f"Final Acc: {correct/total_samples:.4f}")
