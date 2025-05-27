import torch
import tonic
import numpy as np
from tonic import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from f_quant_net import ThreeConvPoolingNet
import time

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
cache_path_test = "./data/nmnist_test_cache"
testset = tonic.datasets.NMNIST(save_to=data_path, transform=frame_transform, train=False)
disk_cached_testset = tonic.DiskCachedDataset(
    testset,
    cache_path=cache_path_test,
    reset_cache=False  # No need to reset if already cached
)
test_loader = DataLoader(
    disk_cached_testset,
    batch_size=64,
    shuffle=False,
    collate_fn=tonic.collation.PadTensors(),
    num_workers=6,
)

# Define symmetric quantization function
def symmetric_quantize_weights(model, bits=8):
    """Apply symmetric Post-Training Quantization to model weights"""
    quantized_model = ThreeConvPoolingNet().to(device)
    quantized_model.load_state_dict(model.state_dict())
    
    # Dictionary to store quantization stats
    quant_stats = {}
    
    with torch.no_grad():
        # Set quantization range for signed integers
        qmin, qmax = -2**(bits-1), 2**(bits-1) - 1
            
        # Quantize convolutional weights
        for name, module in quantized_model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                # Calculate scaling factor based on absolute max value
                w_absmax = torch.abs(module.weight.data).max()
                scale = (qmax - qmin) / (2 * w_absmax) if w_absmax > 0 else 1.0
                
                # Store original weights for comparison
                w_original = module.weight.data.clone()
                
                # Quantize to integers
                w_int = torch.round(module.weight.data * scale)
                
                # Clamp to ensure values are within range
                w_int = torch.clamp(w_int, qmin, qmax)
                
                # Save integer weights (for FPGA implementation)
                module.register_buffer('weight_int', w_int)
                
                # Dequantize for PyTorch inference (would be removed in FPGA)
                module.weight.data = w_int / scale
                
                # Store quantization parameters
                module.register_buffer('scale_factor', torch.tensor(scale))
                
                # Calculate quantization error
                quant_error = torch.mean(torch.abs(w_original - module.weight.data))
                quant_stats[name] = {
                    'scale': scale,
                    'error': quant_error.item(),
                    'bits': bits
                }
                
        # Quantize parameters in pooling layers
        for name, module in quantized_model.named_modules():
            # Threshold parameters
            if hasattr(module, 'threshold') and isinstance(module.threshold, torch.nn.Parameter):
                thresh_absmax = torch.abs(module.threshold.data).max()
                thresh_scale = (qmax - qmin) / (2 * thresh_absmax) if thresh_absmax > 0 else 1.0
                
                # Store original for comparison
                thresh_original = module.threshold.data.clone()
                
                # Quantize to integers
                thresh_int = torch.round(module.threshold.data * thresh_scale)
                thresh_int = torch.clamp(thresh_int, qmin, qmax)
                
                # Save integer thresholds
                module.register_buffer('threshold_int', thresh_int)
                
                # Dequantize for PyTorch
                module.threshold.data = thresh_int / thresh_scale
                
                # Store parameters
                module.register_buffer('threshold_scale', torch.tensor(thresh_scale))
                
                # Calculate error
                thresh_error = torch.mean(torch.abs(thresh_original - module.threshold.data))
                quant_stats[f"{name}.threshold"] = {
                    'scale': thresh_scale,
                    'error': thresh_error.item(),
                    'bits': bits
                }
            
            # Decay parameters
            if hasattr(module, 'decay') and isinstance(module.decay, torch.nn.Parameter):
                decay_absmax = torch.abs(module.decay.data).max()
                decay_scale = (qmax - qmin) / (2 * decay_absmax) if decay_absmax > 0 else 1.0
                
                # Store original
                decay_original = module.decay.data.clone()
                
                # Quantize
                decay_int = torch.round(module.decay.data * decay_scale)
                decay_int = torch.clamp(decay_int, qmin, qmax)
                
                # Save integer decay values
                module.register_buffer('decay_int', decay_int)
                
                # Dequantize for PyTorch
                module.decay.data = decay_int / decay_scale
                
                # Store parameters
                module.register_buffer('decay_scale', torch.tensor(decay_scale))
                
                # Calculate error
                decay_error = torch.mean(torch.abs(decay_original - module.decay.data))
                quant_stats[f"{name}.decay"] = {
                    'scale': decay_scale,
                    'error': decay_error.item(),
                    'bits': bits
                }
    
    return quantized_model, quant_stats

# Function to evaluate model
def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for frames, targets in data_loader:
            frames = frames.to(device)
            targets = targets.to(device)
            
            outputs = model(frames)
            _, predicted = torch.max(outputs.data, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# Load the trained model
print("Loading original model...")
original_model = ThreeConvPoolingNet().to(device)
# Change to your actual model path
original_model.load_state_dict(torch.load('best_model.pt'))  

# Evaluate original model
print("Evaluating original model...")
original_accuracy = evaluate_model(original_model, test_loader, device)
print(f"Original model accuracy: {original_accuracy:.2f}%")

# Test different bit-widths
bit_widths = [8, 6, 4, 3, 2]
accuracies = []
quant_results = {}

for bits in bit_widths:
    print(f"\nTesting {bits}-bit quantization...")
    start_time = time.time()
    
    # Quantize model
    quantized_model, quant_stats = symmetric_quantize_weights(original_model, bits=bits)
    
    # Evaluate quantized model
    accuracy = evaluate_model(quantized_model, test_loader, device)
    accuracies.append(accuracy)
    
    # Store results
    quant_results[bits] = {
        'accuracy': accuracy,
        'accuracy_drop': original_accuracy - accuracy,
        'stats': quant_stats
    }
    
    end_time = time.time()
    print(f"{bits}-bit quantized model accuracy: {accuracy:.2f}%")
    print(f"Accuracy drop: {original_accuracy - accuracy:.2f}%")
    print(f"Quantization time: {end_time - start_time:.2f} seconds")
    
    # Save model if desired
    if bits == 8:  # Save 8-bit model as example
        torch.save(quantized_model.state_dict(), f'quantized_model_{bits}bit.pt')
        print(f"Saved {bits}-bit quantized model")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(bit_widths, accuracies, 'o-', linewidth=2)
plt.xlabel('Bit Width')
plt.ylabel('Accuracy (%)')
plt.title('Post-Training Quantization: Accuracy vs. Bit Width')
plt.grid(True)
plt.xticks(bit_widths)
plt.savefig('ptq_accuracy.png')
print("\nResults plot saved to 'ptq_accuracy.png'")

# Print summary of quantization results
print("\nQuantization Summary:")
print("-" * 50)
print(f"{'Bit-width':<10}{'Accuracy':<15}{'Accuracy Drop':<15}")
print("-" * 50)
for bits in bit_widths:
    result = quant_results[bits]
    print(f"{bits:<10}{result['accuracy']:.2f}%{' ':<7}{result['accuracy_drop']:.2f}%{' ':<5}")

# Print detailed stats for FPGA implementation
print("\nDetailed stats for 8-bit quantization (for FPGA implementation):")
for param_name, stats in quant_results[8]['stats'].items():
    print(f"{param_name}: Scale = {stats['scale']:.4f}, Error = {stats['error']:.6f}")