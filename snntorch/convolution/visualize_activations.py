import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from f_quant_net import ThreeConvPoolingNet
import tonic
from tonic import transforms
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize network activations over time')
    
    # Model and data parameters
    parser.add_argument('--model', type=str, required=True, help='Path to the model file')
    parser.add_argument('--data-path', type=str, default='./data/nmnist', help='Path to NMNIST dataset')
    parser.add_argument('--cache-path', type=str, default='./data/nmnist_test_cache', help='Path to cache')
    parser.add_argument('--sample-idx', type=int, default=0, help='Index of sample to visualize')
    
    # Visualization parameters
    parser.add_argument('--output-dir', type=str, default='./frames', help='Output directory for frames')
    parser.add_argument('--dpi', type=int, default=200, help='DPI for output images')
    parser.add_argument('--stride', type=int, default=1, help='Timestep stride (1 = every frame, 2 = every other frame, etc.)')
    parser.add_argument('--max-frames', type=int, default=0, help='Maximum number of frames to process (0 = all)')
    
    return parser.parse_args()

def process_single_timestep(model, frame, t, device):
    """Process a single timestep and capture activations"""
    with torch.no_grad():
        # Extract single timestep
        frame_t = frame[:, t:t+1, :, :, :].to(device)
        
        # Convert to float tensor
        frame_t = frame_t.float()
        
        # Create membrane potentials
        mem1 = torch.zeros(1, model.conv1.out_channels, 32, 32, device=device)
        mem2 = torch.zeros(1, model.conv2.out_channels, 16, 16, device=device)
        mem3 = torch.zeros(1, model.conv3.out_channels, 8, 8, device=device)
        
        # Process timestep
        xt = frame_t.squeeze(1)
        
        # First layer
        conv1_out = model.conv1(xt)
        mem1, spikes1 = model.pool1(mem1, conv1_out)
        
        # Second layer
        conv2_out = model.conv2(spikes1)
        mem2, spikes2 = model.pool2(mem2, conv2_out)
        
        # Third layer
        conv3_out = model.conv3(spikes2)
        mem3, spikes3 = model.pool3(mem3, conv3_out)
        
        # Classification
        logits = model.classifier(spikes3)
        
        # Get prediction
        pred = torch.argmax(logits, dim=1).item()
        
        # Get class scores
        scores = logits.squeeze().cpu().numpy()
        
        # Return all needed data for visualization
        return {
            'input': xt.squeeze().cpu().numpy(),  # Remove all singleton dimensions
            'spikes1': spikes1.cpu().numpy(),
            'spikes2': spikes2.cpu().numpy(),
            'spikes3': spikes3.cpu().numpy(),
            'logits': scores,
            'prediction': pred
        }

def create_simple_visualization(data, timestep, total_timesteps, output_path, dpi=200):
    """Create a simple visualization with just the most important elements"""
    # Create a simple figure with 3 rows, 2 columns
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    
    # First row: Input image and first feature map from layer 1
    axs[0, 0].imshow(data['input'], cmap='gray')
    axs[0, 0].set_title("Input Sample")
    axs[0, 0].axis('off')
    
    # Show first feature map from each layer
    axs[0, 1].imshow(data['spikes1'][0, 0], cmap='hot')
    axs[0, 1].set_title("Conv1 - First Feature Map")
    axs[0, 1].axis('off')
    
    axs[1, 0].imshow(data['spikes2'][0, 0], cmap='hot')
    axs[1, 0].set_title("Conv2 - First Feature Map")
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(data['spikes3'][0, 0], cmap='hot')
    axs[1, 1].set_title("Conv3 - First Feature Map")
    axs[1, 1].axis('off')
    
    # Class scores
    axs[2, 0].bar(range(10), data['logits'])
    axs[2, 0].set_title("Class Scores")
    axs[2, 0].set_xlabel("Digit")
    axs[2, 0].set_ylabel("Score")
    axs[2, 0].set_xticks(range(10))
    
    # Additional info
    axs[2, 1].axis('off')
    axs[2, 1].text(0.5, 0.7, f"Prediction: {data['prediction']}", 
                  fontsize=14, ha='center',
                  bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    axs[2, 1].text(0.5, 0.3, f"Timestep: {timestep+1}/{total_timesteps}", 
                  fontsize=12, ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)

def load_test_sample(data_path, cache_path, sample_idx=0):
    """Load a test sample from NMNIST dataset"""
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
    testset = tonic.datasets.NMNIST(save_to=data_path, transform=frame_transform, train=False)
    disk_cached_testset = tonic.DiskCachedDataset(
        testset,
        cache_path=cache_path,
        reset_cache=False
    )
    
    # Get specific sample
    frames, target = disk_cached_testset[sample_idx]
    
    # Convert to torch tensor if needed
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    
    # Make sure it's float tensor
    frames = frames.float()
    
    # Add batch dimension
    frames = frames.unsqueeze(0)
    
    return frames, target

def main():
    args = parse_arguments()
    
    # Load model
    print(f"Loading model from {args.model}")
    model = ThreeConvPoolingNet()
    
    # Load state dict but ignore quantization buffers
    state_dict = torch.load(args.model)
    
    # Filter out quantization-related keys
    filtered_state_dict = {}
    for key in model.state_dict().keys():
        if key in state_dict:
            filtered_state_dict[key] = state_dict[key]
    
    model.load_state_dict(filtered_state_dict)
    print("Model loaded successfully (quantization buffers ignored)")
    model.eval()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load test sample
    print(f"Loading test sample {args.sample_idx}")
    sample, target = load_test_sample(args.data_path, args.cache_path, args.sample_idx)
    print(f"Sample shape: {sample.shape}, Target: {target}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get number of timesteps
    _, T, _, _, _ = sample.shape
    
    # Limit frames if specified
    if args.max_frames > 0:
        max_t = min(T, args.max_frames)
    else:
        max_t = T
    
    # Process selected timesteps
    for t in range(0, max_t, args.stride):
        print(f"Processing timestep {t+1}/{T}")
        
        # Process this timestep
        data = process_single_timestep(model, sample, t, device)
        
        # Create visualization and save to file
        output_path = os.path.join(args.output_dir, f"frame_{t:04d}.png")
        create_simple_visualization(data, t, T, output_path, dpi=args.dpi)
    
    print(f"Processed {max_t//args.stride} frames. Saved to {args.output_dir}")
    print(f"To create a video from these frames, you can use:")
    print(f"ffmpeg -framerate 8 -i {args.output_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4")

if __name__ == "__main__":
    main()