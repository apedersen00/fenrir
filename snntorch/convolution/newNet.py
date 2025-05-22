import tonic
import tonic.transforms as tonic_transforms
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader
from conv_nets import NMNISTProofNet
import torch, time, json
import torch.optim as optim

def cout(msg, color="green", icon=None):
    colors = {
        "black": "\033[30m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    icons = {
        "check": "✔️",
        "error": "❌",
        "info": "ℹ️",
        "warn": "⚠️",
        "star": "⭐",
        "rocket": "🚀",
        "test": "🧪"
    }
    color_code = colors.get(color.lower(), colors["green"])
    icon_str = icons.get(icon, "") + " " if icon and icon in icons else ""
    print(f"{color_code}{icon_str}{msg}{colors['reset']}")

def statmsg(msg, kind="info"):
    mapping = {
        "error": {"color": "red", "icon": "error"},
        "success": {"color": "green", "icon": "check"},
        "info": {"color": "blue", "icon": "info"},
        "step": {"color": "yellow", "icon": "star"},
        "warn": {"color": "yellow", "icon": "warn"},
        "test": {"color": "magenta", "icon": "test"},
    }
    opts = mapping.get(kind, {"color": "blue", "icon": "info"})
    cout(msg, color=opts["color"], icon=opts["icon"])
# Parameters
data_path = "./data/nmnist"
cache_path_train = "./cache/nmnist_train"
cache_path_test = "./cache/nmnist_test"
batch_size = 16  # Small for testing, can increase later
time_bin_width_us = 1000

# Define transform: to frames, only keep positive spikes
class OnlyPositive(object):
    def __call__(self, frames):
        # Input: frames [T, 2, H, W] (polarity in dim=1)
        # Output: only positive polarity (assumes polarity index 1 is positive in tonic NMNIST)
        return frames[:, 1:2, :, :]

sensor_size = tonic.datasets.NMNIST.sensor_size

frame_transform = tonic_transforms.Compose([
    tonic_transforms.ToFrame(sensor_size=sensor_size, time_window=time_bin_width_us),
    OnlyPositive()
])
# Load NMNIST dataset
statmsg("Loading NMNIST dataset...", kind="info")
try:
    statmsg("Loading training set...", kind="info")
    trainset = tonic.datasets.NMNIST(save_to=data_path, transform=frame_transform, train=True)
    statmsg("Training set loaded successfully.", kind="success")
except Exception as e:
    statmsg(f"Failed to load training set: {e}", kind="error")

try:
    statmsg("Loading test set...", kind="info")
    testset = tonic.datasets.NMNIST(save_to=data_path, transform=frame_transform, train=False)
    statmsg("Test set loaded successfully.", kind="success")
except Exception as e:
    statmsg(f"Failed to load test set: {e}", kind="error")

statmsg("Caching training set on disk (may take a while first time)...", kind="info")
disk_cached_trainset = DiskCachedDataset(trainset, cache_path=cache_path_train)
statmsg("Caching test set on disk (may take a while first time)...", kind="info")
disk_cached_testset = DiskCachedDataset(testset, cache_path=cache_path_test)
statmsg("Making DataLoaders...", kind="info")
trainloader = DataLoader(
    disk_cached_trainset,
    batch_size=batch_size,
    collate_fn=tonic.collation.PadTensors(),
    shuffle=True,
    num_workers=6
)
testloader = DataLoader(
    disk_cached_testset,
    batch_size=batch_size,
    collate_fn=tonic.collation.PadTensors(),
    num_workers=4
)

# --- TEST: Print a sample batch ---
statmsg("Testing DataLoader...", kind="info")
for i, (frames, targets) in enumerate(trainloader):
    statmsg(f"Batch {i}:", kind="step")
    statmsg(f"  Frames shape: {frames.shape}", kind="test")   # [B, T, 1, H, W]
    statmsg(f"  Targets: {targets}", kind="test")
    if i > 1: break

statmsg(f"Using NMNIST dataset with time bins of {time_bin_width_us} us", kind="info")
statmsg(f"Train set size: {len(trainloader.dataset)}", kind="info")
statmsg(f"Test set size: {len(testloader.dataset)}", kind="info")
statmsg("Using positive spikes only (polarity index 1)", kind="info")

statmsg("Creating model...", kind="info")
model = NMNISTProofNet()
statmsg("Model created.", kind="success")
statmsg(str(model), kind="step")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
statmsg(f"Using device: {device}", kind="info")
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
statmsg(f"Loss function: {criterion}", kind="info")

optimizer = optim.Adam(model.parameters(), lr=1e-3)
statmsg(f"Optimizer: {optimizer}", kind="step")

NUM_EPOCHS = 1
statmsg(f"Training for {NUM_EPOCHS} epochs...", kind="info")

stats = {'train_loss': [], 'train_acc': [], 'test_acc': []}

def train(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    running_loss, running_correct, running_total = 0, 0, 0
    t0 = time.time()
    for batch_idx, (frames, targets) in enumerate(loader):
        frames, targets = frames.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Stats
        running_loss += loss.item() * frames.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == targets).sum().item()
        running_total += frames.size(0)
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            acc = (preds == targets).float().mean().item()
            statmsg(f"Epoch [{epoch}/{total_epochs}] Batch [{batch_idx+1}/{len(loader)}] "
                    f"Loss: {loss.item():.4f} | Batch Acc: {acc:.4f}", kind="info")

    avg_loss = running_loss / running_total
    avg_acc = running_correct / running_total
    statmsg(f"Epoch {epoch} Summary: Avg Loss = {avg_loss:.4f}, Avg Accuracy = {avg_acc:.4f} | "
            f"Epoch time: {time.time() - t0:.2f}s", kind="success")
    return avg_loss, avg_acc

def test(model, loader, device):
    model.eval()
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for frames, targets in loader:
            frames, targets = frames.to(device), targets.to(device)
            outputs = model(frames)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += frames.size(0)
    acc = total_correct / total_samples
    statmsg(f"Test Accuracy: {acc:.4f}", kind="info")
    return acc

statmsg("Starting training loop...", kind="info")
for epoch in range(1, NUM_EPOCHS + 1):
    loss, acc = train(model, trainloader, criterion, optimizer, device, epoch, NUM_EPOCHS)
    stats['train_loss'].append(loss)
    stats['train_acc'].append(acc)
    test_acc = test(model, testloader, device)
    stats['test_acc'].append(test_acc)
    with open('training_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

statmsg("Training loop completed.", kind="success")