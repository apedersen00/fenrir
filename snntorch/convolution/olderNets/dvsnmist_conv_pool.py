import tonic
import tonic.transforms as tonic_transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tonic import MemoryCachedDataset
from tonic import DiskCachedDataset
import json
import time

# 1. Settings
data_path = "./data/nmnist"
batch_size = 128
EPOCHS = 5
time_bin_width_us = 1000
stats_json = "training_stats.json"

class OnlyPositive(object):
    def __call__(self, frames):
        return frames[:, 1:2, :, :]

sensor_size = tonic.datasets.NMNIST.sensor_size

frame_transform = tonic_transforms.Compose([
    tonic_transforms.ToFrame(sensor_size=sensor_size, time_window=time_bin_width_us),
    OnlyPositive()
])

print("Loading datasets...")
trainset = tonic.datasets.NMNIST(save_to=data_path, transform=frame_transform, train=True)
testset = tonic.datasets.NMNIST(save_to=data_path, transform=frame_transform, train=False)

#ram_cached_trainset = MemoryCachedDataset(trainset)
#ram_cached_testset = MemoryCachedDataset(testset)  # <--- memory cache testset too

disk_cached_trainset = DiskCachedDataset(trainset, cache_path="./cache/trainset_cache")
disk_cached_testset = DiskCachedDataset(testset, cache_path="./cache/testset_cache")  # <--- disk cache testset too

trainloader = DataLoader(disk_cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(), shuffle=True, num_workers=8)
testloader = DataLoader(disk_cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())

class SumPooling2D(nn.Module):
    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = x.shape
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        window_area = self.kernel_size ** 2
        num_windows = x_unfold.shape[-1]
        x_unfold = x_unfold.view(batch_size, num_channels, window_area, num_windows)
        pooled_sums = x_unfold.sum(dim=2)
        height_out = (height - self.kernel_size) // self.stride + 1
        width_out = (width - self.kernel_size) // self.stride + 1
        pooled_sums_2d = pooled_sums.view(batch_size, num_channels, height_out, width_out)
        return pooled_sums_2d

class Net(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_classes: int,
            kernel_size_conv: int,
            kernel_size_pool: int,
            stride_pool: int,
            pool_threshold: float,
            neuron_threshold: float,
            neuron_reset_value: float,
            decay: float
    ): 
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=kernel_size_conv,
            stride=1,
            padding=0
        )
        self.sum_pool = SumPooling2D(kernel_size=kernel_size_pool, stride=stride_pool)
        self.neuron_threshold = neuron_threshold
        self.pool_threshold = pool_threshold
        self.decay = decay
        self.neuron_reset_value = neuron_reset_value
    
    def forward(self, input_sequence):
        batch_size, num_steps, _, height, width = input_sequence.shape
        device = input_sequence.device
        conv_out = self.conv(input_sequence[:, 0])
        _, num_classes, conv_h, conv_w = conv_out.shape
        membrane = torch.zeros(batch_size, num_classes, conv_h, conv_w, device=device)
        spike_trains = []
        for t in range(num_steps):
            conv_out = self.conv(input_sequence[:, t])
            membrane = membrane - self.decay + conv_out
            pooled_sums = self.sum_pool(membrane)
            spike_trains.append(pooled_sums)
            reset_mask = membrane >= self.neuron_threshold
            membrane = membrane.masked_fill(reset_mask, self.neuron_reset_value)
        spike_trains = torch.stack(spike_trains, dim=0)
        spike_trains = spike_trains.permute(1, 0, 2, 3, 4)
        class_scores = spike_trains.sum(dim=(1, 3, 4))  # [batch, num_classes]
        return class_scores

def train(model, loader, criterion, optimizer, device, epoch, total_epochs, stats):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    batch_times = []
    epoch_start = time.time()

    for batch_idx, (frames, targets) in enumerate(loader):
        batch_start = time.time()

        frames, targets = frames.to(device), targets.to(device)
        optimizer.zero_grad()
        class_scores = model(frames)
        loss = criterion(class_scores, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * frames.size(0)
        preds = class_scores.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += frames.size(0)

        batch_elapsed = time.time() - batch_start
        batch_times.append(batch_elapsed)
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            batch_speed = frames.size(0) / batch_elapsed
            print(f"Epoch [{epoch}/{total_epochs}] Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f} | "
                  f"Batch Acc: {(preds == targets).float().mean():.4f} | "
                  f"Batch time: {batch_elapsed:.3f}s | {batch_speed:.2f} samples/sec")

    epoch_elapsed = time.time() - epoch_start
    avg_batch_time = sum(batch_times) / len(batch_times)
    print(f"Epoch {epoch} Summary: Avg Loss = {total_loss / total_samples:.4f}, "
          f"Avg Accuracy = {total_correct / total_samples:.4f} | "
          f"Epoch time: {epoch_elapsed:.2f}s | Avg batch time: {avg_batch_time:.3f}s")

    stats['train_loss'].append(total_loss / total_samples)
    stats['train_acc'].append(total_correct / total_samples)
    stats.setdefault('train_batch_time', []).append(avg_batch_time)
    stats.setdefault('train_epoch_time', []).append(epoch_elapsed)
    return total_loss / total_samples, total_correct / total_samples


def test(model, loader, device, stats):
    model.eval()
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for frames, targets in loader:
            frames, targets = frames.to(device), targets.to(device)
            class_scores = model(frames)
            preds = class_scores.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += frames.size(0)
    acc = total_correct / total_samples
    print(f"Test Accuracy: {acc:.4f}")
    stats['test_acc'].append(acc)
    return acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = Net(
        in_channels=1,
        out_channels=10,
        num_classes=10,
        kernel_size_conv=3,
        kernel_size_pool=2,
        stride_pool=2,
        pool_threshold=3.0,
        neuron_threshold=1.0,
        neuron_reset_value=0.0,
        decay=0.05
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    stats = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(1, EPOCHS + 1):
        train(model, trainloader, criterion, optimizer, device, epoch, EPOCHS, stats)
        test(model, testloader, device, stats)
        # Save stats every epoch
        with open(stats_json, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"Saved training statistics to {stats_json}")

if __name__ == "__main__":
    main()
