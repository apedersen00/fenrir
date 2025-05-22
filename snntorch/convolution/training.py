import tonic
import tonic.transforms as tonic_transforms
from tonic import DiskCachedDataset
from torch.utils.data import DataLoader
from conv_nets import NMNISTV2
import torch, time, json, os
import torch.optim as optim
import snntorch.functional as SF

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
batch_size = 32  # Small for testing, can increase later
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
model = NMNISTV2()
statmsg("Model created.", kind="success")
statmsg(str(model), kind="step")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
statmsg(f"Using device: {device}", kind="info")
model = model.to(device)

sample_frames, sample_target = trainloader.dataset[0]
sample_frames = torch.from_numpy(sample_frames).float().unsqueeze(0).to(device)  # [1, T, 1, H, W]
with torch.no_grad():
    sample_output = model(sample_frames)
statmsg(f"Sample output shape: {sample_output.shape}", kind="test")
statmsg(f"Sample output: {sample_output}", kind="test")

criterion = torch.nn.CrossEntropyLoss()
statmsg(f"Loss function: {criterion}", kind="info")

optimizer = optim.Adam(model.parameters(), lr=1e-3)
statmsg(f"Optimizer: {optimizer}", kind="step")



stats_dir = "./stats"
batches_dir = stats_dir + "/batches"
if not os.path.exists(stats_dir):
    statmsg(f"Creating stats directory at {stats_dir}...", kind="info")
    os.makedirs(stats_dir)
if not os.path.exists(batches_dir):
    statmsg(f"Creating batches directory at {batches_dir}...", kind="info")
    os.makedirs(batches_dir)

def train(
        model,
        loader,
        criterion,
        optimizer,
        device,
        epoch,
        total_epochs,
        batches_dir
):
    model.train()
    running_loss, running_correct, running_total = 0, 0, 0
    t0 = time.time()
    batch_stats_all = []
    statmsg(f"Epoch {epoch} started at {time.strftime('%H:%M:%S', time.localtime(t0))}", kind="step")

    for batch_idx, (frames, targets) in enumerate(loader):
        frames, targets = frames.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        batch_loss = loss.item()
        batch_acc = (preds == targets).float().mean().item()

        # Collect stats for this batch (every 10th batch, or for the first 10)
        if batch_idx < 10 or batch_idx % 10 == 0:
            batch_stats = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'batch_loss': batch_loss,
                'batch_acc': batch_acc,
                'conv1_stats': model.conv1_stats,
                'pool1_stats': model.pool1_stats,
                'conv2_stats': model.conv2_stats,
                'pool2_stats': model.pool2_stats,
            }
            batch_stats_all.append(batch_stats)

        running_loss += batch_loss * frames.size(0)
        running_correct += (preds == targets).sum().item()
        running_total += frames.size(0)

        # print progress
        if (batch_idx + 1) % 10 == 0:
            avg_acc = running_correct / running_total
            statmsg(f"Epoch [{epoch}/{total_epochs}] Batch [{batch_idx+1}/{len(loader)}] "
                    f"Loss: {batch_loss:.4f} Acc: {batch_acc:.4f} | Running Avg Acc: {avg_acc:.4f}", kind="step")

    # Save all collected batch stats for this epoch to a single file
    batch_stats_file = os.path.join(batches_dir, f"epoch_{epoch}_batches.json")
    with open(batch_stats_file, 'w') as f:
        json.dump(batch_stats_all, f, indent=2)

    avg_loss = running_loss / running_total
    avg_acc = running_correct / running_total
    statmsg(f"Epoch {epoch} Summary: Avg Loss = {avg_loss:.4f}, Avg Accuracy = {avg_acc:.4f}", kind="info")
    return avg_loss, avg_acc



def test(model, loader, device, criterion, epoch=None, batches_dir="./stats/test_batches"):
    model.eval()
    total_correct, total_samples = 0, 0
    batch_stats_all = []
    running_loss = 0

    if epoch is not None and not os.path.exists(batches_dir):
        os.makedirs(batches_dir)

    with torch.no_grad():
        for batch_idx, (frames, targets) in enumerate(loader):
            frames, targets = frames.to(device), targets.to(device)
            outputs = model(frames)
            preds = outputs.argmax(dim=1)
            loss = criterion(outputs, targets)
            batch_loss = loss.item()
            batch_acc = (preds == targets).float().mean().item()

            total_correct += (preds == targets).sum().item()
            total_samples += frames.size(0)
            running_loss += batch_loss * frames.size(0)

            batch_stats = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'batch_loss': batch_loss,
                'batch_acc': batch_acc,
                # Optionally include stats from the model (like conv1_stats, etc.)
            }
            batch_stats_all.append(batch_stats)

            statmsg(f"Testing batch {batch_idx}: Loss: {batch_loss:.4f} Acc: {batch_acc:.4f} | Processed {total_samples} samples so far...", kind="step")

    avg_loss = running_loss / total_samples
    acc = total_correct / total_samples
    statmsg(f"Test Summary: Avg Loss = {avg_loss:.4f}, Accuracy = {acc:.4f}", kind="success")

    # Save test batch stats to file if epoch is provided
    if epoch is not None:
        stats_file = os.path.join(batches_dir, f"test_batches_epoch_{epoch}.json")
        with open(stats_file, 'w') as f:
            json.dump(batch_stats_all, f, indent=2)

    return acc, avg_loss


NUM_EPOCHS = 10
statmsg(f"Training for {NUM_EPOCHS} epochs...", kind="info")

train_stats = {'train_loss': [], 'train_acc': [], 'test_acc': []}

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = train(model, trainloader, criterion, optimizer, device, epoch, NUM_EPOCHS, batches_dir)
    test_acc = test(model, testloader, device)

    train_stats['train_loss'].append(train_loss)
    train_stats['train_acc'].append(train_acc)
    train_stats['test_acc'].append(test_acc)

    # Save epoch summary stats
    with open(f"{stats_dir}/training_stats_epoch_{epoch}.json", "w") as f:
        json.dump(train_stats, f, indent=2)

    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_stats': train_stats,
    }
    statmsg(f"Saving checkpoint for epoch {epoch}...", kind="info")
    torch.save(ckpt, f'nmnistv2_epoch{epoch}.pth')