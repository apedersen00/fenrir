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
EPOCHS = 30
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
            conv1_out: int,
            conv2_out: int,
            num_classes: int,
            kernel_size_conv1: int,
            kernel_size_conv2: int,
            kernel_size_pool1: int,
            kernel_size_pool2: int,
            stride_pool1: int,
            stride_pool2: int,
            neuron_threshold: float,
            neuron_reset_value: float,
            decay: float
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=conv1_out,
            kernel_size=kernel_size_conv1,
            stride=1,
            padding=0
        )
        self.pool1 = SumPooling2D(kernel_size=kernel_size_pool1, stride=stride_pool1)
        self.conv2 = nn.Conv2d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=kernel_size_conv2,
            stride=1,
            padding=0
        )
        self.pool2 = SumPooling2D(kernel_size=kernel_size_pool2, stride=stride_pool2)

        # Calculate output size after conv/pool for FC
        # This may need a dummy forward or manual calculation based on your input size
        self.fc_in_features = None  # will set after first forward
        self.fc = None  # Lazy init
        self.num_classes = num_classes

        self.neuron_threshold = neuron_threshold
        self.decay = decay
        self.neuron_reset_value = neuron_reset_value

    def forward(self, input_sequence):
        batch_size, num_steps, _, height, width = input_sequence.shape
        device = input_sequence.device

        # First conv layer output size:
        conv1_h = height - self.conv1.kernel_size[0] + 1
        conv1_w = width - self.conv1.kernel_size[1] + 1

        mem1 = torch.zeros(batch_size, self.conv1.out_channels, conv1_h, conv1_w, device=device)
        mem2 = None  # To be set after pool1
        spike_trains = []

        for t in range(num_steps):
            # Conv1
            conv1_out = self.conv1(input_sequence[:, t])
            mem1 = mem1 - self.decay + conv1_out
            pool1_out = self.pool1(mem1)

            # Set up mem2 on first step
            if mem2 is None:
                p1h, p1w = pool1_out.shape[-2], pool1_out.shape[-1]
                conv2_h = p1h - self.conv2.kernel_size[0] + 1
                conv2_w = p1w - self.conv2.kernel_size[1] + 1
                mem2 = torch.zeros(batch_size, self.conv2.out_channels, conv2_h, conv2_w, device=device)

            # Conv2
            conv2_out = self.conv2(pool1_out)
            mem2 = mem2 - self.decay + conv2_out
            pool2_out = self.pool2(mem2)

            # Collect output
            spike_trains.append(pool2_out)

            # Membrane resets
            mem1 = mem1.masked_fill(mem1 >= self.neuron_threshold, self.neuron_reset_value)
            mem2 = mem2.masked_fill(mem2 >= self.neuron_threshold, self.neuron_reset_value)

        spikes = torch.stack(spike_trains, dim=0)  # [timesteps, batch, c2, H, W]
        spikes = spikes.permute(1, 0, 2, 3, 4)    # [batch, timesteps, c2, H, W]
        summed = spikes.sum(dim=(1, 3, 4))        # [batch, c2]

        # FC setup
        if self.fc is None:
            # If you want time+spatial sum:
            features = spikes.sum(dim=(1, 3, 4, 0))  # [batch, c2] (sum over all but batch, c2)
            in_features = summed.shape[1]
            self.fc = nn.Linear(in_features, self.num_classes).to(device)
        out = self.fc(summed)
        return out


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


def test(model, loader, device, stats, epoch=None, total_epochs=None):
    model.eval()
    total_correct, total_samples = 0, 0
    batch_count = len(loader)
    with torch.no_grad():
        for batch_idx, (frames, targets) in enumerate(loader):
            frames, targets = frames.to(device), targets.to(device)
            class_scores = model(frames)
            preds = class_scores.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += frames.size(0)
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == batch_count:
                print(f" [Test] Batch [{batch_idx+1}/{batch_count}] Acc: {(preds == targets).float().mean():.4f}")
    acc = total_correct / total_samples
    print(f"Test Accuracy (epoch {epoch if epoch is not None else ''}): {acc:.4f}")
    stats['test_acc'].append(acc)
    return acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = Net(
        in_channels=1,
        conv1_out=10,
        conv2_out=10,
        num_classes=10,
        kernel_size_conv1=3,
        kernel_size_conv2=3,
        kernel_size_pool1=2,
        kernel_size_pool2=2,
        stride_pool1=2,
        stride_pool2=2,
        neuron_threshold=5.0,
        neuron_reset_value=0.0,
        decay=0.05
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    stats = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(1, EPOCHS + 1):
        train(model, trainloader, criterion, optimizer, device, epoch, EPOCHS, stats)
        test(model, testloader, device, stats, epoch=epoch, total_epochs=EPOCHS)

        # Save stats every epoch
        with open(stats_json, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"Saved training statistics to {stats_json}")

        # Save model checkpoint every 5 epochs
        if epoch % 5 == 0 or epoch == EPOCHS:
            checkpoint_path = f"model_epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    main()