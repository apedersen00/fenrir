import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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

def process_timesteps(model, sample, device, max_t, stride=1):
    """Process all timesteps and track membrane potentials"""
    B, T, C, H, W = sample.shape
    
    # Track all activations and accumulated logits
    all_data = []
    accumulated_logits = torch.zeros(10, device=device)
    
    for t in range(0, max_t, stride):
        print(f"Processing timestep {t+1}/{T}")
        
        # Extract single timestep
        frame_t = sample[:, t:t+1, :, :, :].to(device)
        frame_t = frame_t.float()
        
        # Create membrane potentials
        mem1 = torch.zeros(1, model.conv1.out_channels, 32, 32, device=device)
        mem2 = torch.zeros(1, model.conv2.out_channels, 16, 16, device=device)
        mem3 = torch.zeros(1, model.conv3.out_channels, 8, 8, device=device)
        
        # Process timestep
        xt = frame_t.squeeze(1)
        
        # Forward pass
        with torch.no_grad():
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
            
            # Accumulate logits
            accumulated_logits += logits.squeeze()
            
            # Get prediction from accumulated logits
            pred = torch.argmax(accumulated_logits).item()
            
            # Store data for this timestep
            timestep_data = {
                'input': xt.squeeze().cpu().numpy(),
                'mem1': mem1.cpu().numpy(),  # Store membrane potentials
                'mem2': mem2.cpu().numpy(),
                'mem3': mem3.cpu().numpy(),
                'spikes1': spikes1.cpu().numpy(),
                'spikes2': spikes2.cpu().numpy(),
                'spikes3': spikes3.cpu().numpy(),
                'logits': logits.squeeze().cpu().numpy(),
                'accumulated_logits': accumulated_logits.cpu().numpy(),
                'prediction': pred,
                'timestep': t
            }
            
            all_data.append(timestep_data)
    
    return all_data

def process_timesteps(model, sample, device, max_t, stride=1):
    """Process all timesteps and track membrane potentials"""
    B, T, C, H, W = sample.shape
    
    # Track all activations and accumulated logits
    all_data = []
    accumulated_logits = torch.zeros(10, device=device)
    
    # Create persistent membrane potentials
    mem1 = torch.zeros(1, model.conv1.out_channels, 32, 32, device=device)
    mem2 = torch.zeros(1, model.conv2.out_channels, 16, 16, device=device)
    mem3 = torch.zeros(1, model.conv3.out_channels, 8, 8, device=device)
    
    for t in range(0, max_t, stride):
        print(f"Processing timestep {t+1}/{T}")
        
        # Extract single timestep
        frame_t = sample[:, t:t+1, :, :, :].to(device)
        frame_t = frame_t.float()
        
        # Process timestep
        xt = frame_t.squeeze(1)
        
        # Forward pass
        with torch.no_grad():
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
            
            # Accumulate logits
            accumulated_logits += logits.squeeze()
            
            # Get prediction from accumulated logits
            pred = torch.argmax(accumulated_logits).item()
            
            # Store data for this timestep
            timestep_data = {
                'input': xt.squeeze().cpu().numpy(),
                'mem1': mem1.cpu().numpy().copy(),  # Make sure to copy
                'mem2': mem2.cpu().numpy().copy(),
                'mem3': mem3.cpu().numpy().copy(),
                'spikes1': spikes1.cpu().numpy(),
                'spikes2': spikes2.cpu().numpy(),
                'spikes3': spikes3.cpu().numpy(),
                'logits': logits.squeeze().cpu().numpy(),
                'accumulated_logits': accumulated_logits.cpu().numpy(),
                'prediction': pred,
                'timestep': t
            }
            
            all_data.append(timestep_data)
    
    return all_data

def create_custom_visualization(data, timestep, total_timesteps, target_digit, output_path, dpi=200):
    """Create visualization with membrane potentials as heatmaps"""
    fig = plt.figure(figsize=(18, 10))
    
    # Add title showing the target digit
    fig.suptitle(f"NMNIST Sample - Target Digit: {target_digit}", fontsize=16)
    
    # Define main grid layout to match the sketch
    main_grid = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.5], 
                        width_ratios=[1, 1.5, 1], hspace=0.3, wspace=0.3)
    
    # Box 1: Input sample (top-left)
    ax1 = fig.add_subplot(main_grid[0, 0])
    ax1.imshow(data['input'], cmap='gray')
    ax1.set_title("Input Sample")
    ax1.axis('off')
    
    # Find global min/max for each layer for consistent colormaps
    mem1_min, mem1_max = np.min(data['mem1']), np.max(data['mem1'])
    mem2_min, mem2_max = np.min(data['mem2']), np.max(data['mem2'])
    mem3_min, mem3_max = np.min(data['mem3']), np.max(data['mem3'])
    
    # Add some margin to ensure values aren't at colormap extremes
    def add_margin(min_val, max_val, margin=0.1):
        range_val = max_val - min_val
        if range_val < 1e-6:  # Avoid division by very small values
            return min_val - 0.1, max_val + 0.1
        return min_val - margin * range_val, max_val + margin * range_val
    
    mem1_min, mem1_max = add_margin(mem1_min, mem1_max)
    mem2_min, mem2_max = add_margin(mem2_min, mem2_max)
    mem3_min, mem3_max = add_margin(mem3_min, mem3_max)
    
    # Box 2: Conv1 membrane potentials (top-middle)
    ax2 = fig.add_subplot(main_grid[0, 1])
    ax2.set_title("Conv1 Membrane Potentials")
    ax2.axis('off')
    
    # Create a grid for conv1 feature maps
    n_maps1 = min(data['mem1'].shape[1], 12)
    n_cols1 = 4
    n_rows1 = int(np.ceil(n_maps1 / n_cols1))
    
    # Inner grid for conv1
    gs1 = GridSpec(n_rows1, n_cols1, figure=fig, 
                  left=ax2.get_position().x0, right=ax2.get_position().x1,
                  bottom=ax2.get_position().y0, top=ax2.get_position().y1)
    
    # Plot conv1 membrane potentials as heatmaps
    for i in range(n_maps1):
        ax = fig.add_subplot(gs1[i // n_cols1, i % n_cols1])
        im = ax.imshow(data['mem1'][0, i], cmap='plasma', vmin=mem1_min, vmax=mem1_max)
        
        # Add spike markers (white dots where spikes occur)
        if np.max(data['spikes1'][0, i]) > 0:  # If there are spikes
            y, x = np.where(data['spikes1'][0, i] > 0.5)
            ax.scatter(x, y, s=10, c='white', marker='o')
            
        # Add small colorbar to show the potential range
        if i == n_maps1 - 1:  # Add colorbar only for the last map
            cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7)
            cbar.set_label('Potential')
        ax.axis('off')
    
    # Box 4: Conv3 membrane potentials (top-right)
    ax4 = fig.add_subplot(main_grid[0, 2])
    ax4.set_title("Conv3 Membrane Potentials")
    ax4.axis('off')
    
    # Create a grid for conv3 feature maps
    n_maps3 = min(data['mem3'].shape[1], 10)
    n_cols3 = 3
    n_rows3 = int(np.ceil(n_maps3 / n_cols3))
    
    # Inner grid for conv3
    gs3 = GridSpec(n_rows3, n_cols3, figure=fig, 
                  left=ax4.get_position().x0, right=ax4.get_position().x1,
                  bottom=ax4.get_position().y0, top=ax4.get_position().y1)
    
    # Plot conv3 membrane potentials as heatmaps
    for i in range(n_maps3):
        ax = fig.add_subplot(gs3[i // n_cols3, i % n_cols3])
        im = ax.imshow(data['mem3'][0, i], cmap='plasma', vmin=mem3_min, vmax=mem3_max)
        
        # Add spike markers
        if np.max(data['spikes3'][0, i]) > 0:  # If there are spikes
            y, x = np.where(data['spikes3'][0, i] > 0.5)
            ax.scatter(x, y, s=10, c='white', marker='o')
            
        # Add small colorbar to the last map
        if i == n_maps3 - 1:
            cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7)
            cbar.set_label('Potential')
        ax.axis('off')
    
    # Box 3: Conv2 membrane potentials (bottom spanning 2 columns)
    ax3 = fig.add_subplot(main_grid[1, 0:2])
    ax3.set_title("Conv2 Membrane Potentials")
    ax3.axis('off')
    
    # Create a grid for conv2 feature maps
    n_maps2 = min(data['mem2'].shape[1], 24)
    n_cols2 = 6
    n_rows2 = int(np.ceil(n_maps2 / n_cols2))
    
    # Inner grid for conv2
    gs2 = GridSpec(n_rows2, n_cols2, figure=fig, 
                  left=ax3.get_position().x0, right=ax3.get_position().x1,
                  bottom=ax3.get_position().y0, top=ax3.get_position().y1)
    
    # Plot conv2 membrane potentials as heatmaps
    for i in range(n_maps2):
        ax = fig.add_subplot(gs2[i // n_cols2, i % n_cols2])
        im = ax.imshow(data['mem2'][0, i], cmap='plasma', vmin=mem2_min, vmax=mem2_max)
        
        # Add spike markers
        if np.max(data['spikes2'][0, i]) > 0:  # If there are spikes
            y, x = np.where(data['spikes2'][0, i] > 0.5)
            ax.scatter(x, y, s=10, c='white', marker='o')
            
        # Add small colorbar to the last map
        if i == n_maps2 - 1:
            cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.7)
            cbar.set_label('Potential')
        ax.axis('off')
    
    # Box 5: Accumulated class scores bar plot (bottom-right)
    ax5 = fig.add_subplot(main_grid[1, 2])
    
    # Color the bars - highlight the target and current prediction
    bar_colors = ['#1f77b4'] * 10  # Default color
    bar_colors[target_digit] = '#2ca02c'  # Green for target
    if data['prediction'] != target_digit:
        bar_colors[data['prediction']] = '#d62728'  # Red for wrong prediction
    
    ax5.bar(range(10), data['accumulated_logits'], color=bar_colors)
    ax5.set_title("Accumulated Class Scores")
    ax5.set_xlabel("Digit")
    ax5.set_ylabel("Score")
    ax5.set_xticks(range(10))
    
    # Add legend for bar colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', label='Target'),
    ]
    if data['prediction'] != target_digit:
        legend_elements.append(Patch(facecolor='#d62728', label='Current Prediction'))
    ax5.legend(handles=legend_elements, loc='upper right')
    
    # Box 6: Prediction (overlay on bottom-right)
    prediction_text = f"Prediction: {data['prediction']}"
    fig.text(0.85, 0.15, prediction_text, fontsize=14, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    # Add timestep counter
    fig.text(0.02, 0.02, f"Timestep: {data['timestep']+1}/{total_timesteps}", fontsize=12)
    
    # Save figure
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
    
    # Process all timesteps
    all_timestep_data = process_timesteps(model, sample, device, max_t, args.stride)
    
    # In the main function, replace the visualization loop with this:
    total_frames = len(all_timestep_data)
    print(f"Processing complete. Now creating {total_frames} visualizations...")

    for i, data in enumerate(all_timestep_data):
        print(f"Creating visualization {i+1}/{total_frames}")
        output_path = os.path.join(args.output_dir, f"frame_{i:04d}.png")
        create_custom_visualization(data, i, total_frames, target, output_path, dpi=args.dpi)
    
    print(f"Processed {len(all_timestep_data)} frames. Saved to {args.output_dir}")
    print(f"To create a video from these frames, you can use:")
    print(f"ffmpeg -framerate 8 -i {args.output_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4")

if __name__ == "__main__":
    main()