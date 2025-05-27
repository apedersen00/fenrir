import torch
import tonic
import numpy as np
from tonic import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from f_quant_net import ThreeConvPoolingNet  # Import your model

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data transformations
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

# Load test dataset
data_path = "./data/nmnist"
cache_path_test = "./data/nmnist_test_cache"  # Changed from train to test cache
testset = tonic.datasets.NMNIST(save_to=data_path, transform=frame_transform, train=False)
disk_cached_testset = tonic.DiskCachedDataset(
    testset,
    cache_path=cache_path_test,
    reset_cache=True
)
test_loader = DataLoader(
    disk_cached_testset,
    batch_size=64,
    shuffle=False,
    collate_fn=tonic.collation.PadTensors(),
    num_workers=6,
)

# Load the trained model
model = ThreeConvPoolingNet().to(device)
model.load_state_dict(torch.load('model_finetuned.pt'))  # Or 'best_model.pt' if not finetuned
model.eval()  # Set to evaluation mode

# Test the model
total = 0
correct = 0
all_preds = []
all_targets = []

with torch.no_grad():  # No gradient calculation needed for testing
    for frames, targets in test_loader:
        frames = frames.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(frames)
        
        # Get predictions
        _, predicted = torch.max(outputs.data, 1)
        
        # Update statistics
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        # Save predictions and targets for confusion matrix
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
        # Print progress
        if total % 1000 == 0:
            print(f"Processed {total} test samples")

# Calculate accuracy
accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

# Create confusion matrix
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f}%)')
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved to 'confusion_matrix.png'")

# Per-class accuracy
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for frames, targets in test_loader:
        frames = frames.to(device)
        targets = targets.to(device)
        outputs = model(frames)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == targets).squeeze()
        for i in range(targets.size(0)):
            label = targets[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# Print per-class accuracy
for i in range(10):
    print(f'Accuracy of digit {i}: {100 * class_correct[i] / class_total[i]:.2f}%')

# Plot misclassified examples (optional)
def plot_misclassified(model, test_loader, device, num_samples=5):
    misclassified_frames = []
    misclassified_preds = []
    misclassified_targets = []
    
    model.eval()
    with torch.no_grad():
        for frames, targets in test_loader:
            frames = frames.to(device)
            targets = targets.to(device)
            outputs = model(frames)
            _, preds = torch.max(outputs, 1)
            
            # Find misclassified examples
            incorrect_mask = preds != targets
            if incorrect_mask.sum() > 0:
                misclassified_idx = torch.where(incorrect_mask)[0]
                for idx in misclassified_idx:
                    misclassified_frames.append(frames[idx].cpu())
                    misclassified_preds.append(preds[idx].item())
                    misclassified_targets.append(targets[idx].item())
                    
                    if len(misclassified_frames) >= num_samples:
                        break
            
            if len(misclassified_frames) >= num_samples:
                break
    
    # Plot misclassified examples
    fig, axs = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        # Sum across time dimension for visualization
        frame_sum = misclassified_frames[i][0].sum(dim=0)
        axs[i].imshow(frame_sum, cmap='hot')
        axs[i].set_title(f'True: {misclassified_targets[i]}\nPred: {misclassified_preds[i]}')
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('misclassified_examples.png')
    print("Misclassified examples saved to 'misclassified_examples.png'")

# Uncomment to plot misclassified examples
plot_misclassified(model, test_loader, device)