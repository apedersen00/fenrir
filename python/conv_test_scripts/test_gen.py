#!/usr/bin/env python3
"""
Enhanced SNN Event Generator for SystemVerilog Testbench and PyTorch
Generates event files for Vivado simulation and PyTorch-compatible data for comparison
"""

import argparse
import sys
import random
import math
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def parse_arguments():
    """Parse command line arguments for SNN event generation"""
    parser = argparse.ArgumentParser(
        description='Generate SNN event files and PyTorch-compatible data for testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 2 4 100 8 8 7
  %(prog)s 1 2 50 16 16 8 --timestep_idx 25 --torch_output
  %(prog)s 3 6 200 32 32 8 --output_dir test_data --prefix my_test --torch_output
        """
    )
    
    # Required positional arguments
    parser.add_argument('in_channels', type=int, 
                       help='Number of input channels')
    parser.add_argument('out_channels', type=int, 
                       help='Number of output channels')
    parser.add_argument('num_events', type=int, 
                       help='Number of events to generate')
    parser.add_argument('img_w', type=int, 
                       help='Image width (pixels)')
    parser.add_argument('img_h', type=int, 
                       help='Image height (pixels)')
    parser.add_argument('bits_per_coord', type=int, 
                       help='Number of bits per coordinate (x,y)')
    
    # Optional arguments
    parser.add_argument('--timestep_idx', type=int, default=None,
                       help='Index where to insert timestep event (0-based). If omitted, no timestep events are generated.')
    
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for generated files (default: current directory)')
    
    parser.add_argument('--prefix', type=str, default='snn_test',
                       help='Prefix for output filenames (default: snn_test)')
    
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible event generation')
    
    parser.add_argument('--pattern', type=str, choices=['random', 'cross', 'corners', 'grid'], 
                       default='random',
                       help='Event generation pattern (default: random)')
    
    # PyTorch-specific arguments
    parser.add_argument('--torch_output', action='store_true',
                       help='Generate PyTorch-compatible files (.npy and .json)')
    
    parser.add_argument('--accumulation_mode', type=str, choices=['binary', 'count', 'weighted'],
                       default='binary',
                       help='How to accumulate spikes into feature maps (default: binary)')
    
    parser.add_argument('--time_windows', type=int, default=1,
                       help='Number of time windows for PyTorch data (default: 1)')
    
    parser.add_argument('--normalize_torch', action='store_true',
                       help='Normalize PyTorch tensors to [0,1] range')
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate the parsed arguments"""
    errors = []
    
    # Check positive values
    if args.in_channels <= 0:
        errors.append("in_channels must be positive")
    if args.out_channels <= 0:
        errors.append("out_channels must be positive")
    if args.num_events <= 0:
        errors.append("num_events must be positive")
    if args.img_w <= 0:
        errors.append("img_w must be positive")
    if args.img_h <= 0:
        errors.append("img_h must be positive")
    if args.bits_per_coord <= 0:
        errors.append("bits_per_coord must be positive")
    
    # Check reasonable limits
    if args.in_channels > 32:
        errors.append("in_channels should be <= 32 (limited by practical spike vector size)")
    if args.bits_per_coord > 16:
        errors.append("bits_per_coord should be <= 16 (limited by practical coordinate range)")
    
    # Check coordinate range vs image size
    max_coord_val = (1 << args.bits_per_coord) - 1
    if args.img_w > max_coord_val + 1:
        errors.append(f"img_w ({args.img_w}) exceeds maximum coordinate value ({max_coord_val}) for {args.bits_per_coord} bits")
    if args.img_h > max_coord_val + 1:
        errors.append(f"img_h ({args.img_h}) exceeds maximum coordinate value ({max_coord_val}) for {args.bits_per_coord} bits")
    
    # Check timestep index
    if args.timestep_idx is not None:
        if args.timestep_idx < 0:
            errors.append("timestep_idx must be non-negative")
        if args.timestep_idx >= args.num_events:
            errors.append(f"timestep_idx ({args.timestep_idx}) must be less than num_events ({args.num_events})")
    
    # Check PyTorch-specific arguments
    if args.time_windows <= 0:
        errors.append("time_windows must be positive")
    
    return errors


def generate_events(args):
    """Generate events based on the specified pattern"""
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    events = []
    
    if args.pattern == 'random':
        events = generate_random_events(args)
    elif args.pattern == 'cross':
        events = generate_cross_pattern(args)
    elif args.pattern == 'corners':
        events = generate_corners_pattern(args)
    elif args.pattern == 'grid':
        events = generate_grid_pattern(args)
    
    return events


def generate_random_events(args):
    """Generate random events"""
    events = []
    
    for i in range(args.num_events):
        # Determine if this is a timestep event
        is_timestep = (args.timestep_idx is not None and i == args.timestep_idx)
        
        # Generate coordinates within image bounds
        x = random.randint(0, args.img_w - 1)
        y = random.randint(0, args.img_h - 1)
        
        # Generate spike pattern
        if is_timestep:
            spikes = 0  # Timestep events have no spikes
        else:
            # Generate spike pattern with at least one spike guaranteed
            spikes = 0
            
            # First, guarantee at least one spike by randomly selecting one channel
            guaranteed_channel = random.randint(0, args.in_channels - 1)
            spikes |= (1 << guaranteed_channel)
            
            # Then add additional spikes with ~30% probability per remaining channel
            for ch in range(args.in_channels):
                if ch != guaranteed_channel and random.random() < 0.3:
                    spikes |= (1 << ch)
        
        event = {
            'timestep': 1 if is_timestep else 0,
            'x': x,
            'y': y,
            'spikes': spikes
        }
        events.append(event)
    
    return events


def generate_cross_pattern(args):
    """Generate cross pattern in center of image"""
    events = []
    center_x = args.img_w // 2
    center_y = args.img_h // 2
    
    # Horizontal line
    for x in range(max(0, center_x - 2), min(args.img_w, center_x + 3)):
        event = {
            'timestep': 0,
            'x': x,
            'y': center_y,
            'spikes': 1  # Channel 0 active (guaranteed at least one spike)
        }
        events.append(event)
    
    # Vertical line (skip center to avoid duplicate)
    for y in range(max(0, center_y - 2), min(args.img_h, center_y + 3)):
        if y != center_y:
            event = {
                'timestep': 0,
                'x': center_x,
                'y': y,
                'spikes': 1  # Channel 0 active (guaranteed at least one spike)
            }
            events.append(event)
    
    # Add additional random events to reach desired count
    while len(events) < args.num_events:
        x = random.randint(0, args.img_w - 1)
        y = random.randint(0, args.img_h - 1)
        # Skip if this coordinate already exists
        if not any(e['x'] == x and e['y'] == y for e in events):
            # Guarantee at least one spike
            guaranteed_channel = random.randint(0, args.in_channels - 1)
            spikes = 1 << guaranteed_channel
            
            # Add additional spikes with probability
            for ch in range(args.in_channels):
                if ch != guaranteed_channel and random.random() < 0.3:
                    spikes |= (1 << ch)
                    
            event = {
                'timestep': 0,
                'x': x,
                'y': y,
                'spikes': spikes
            }
            events.append(event)
    
    # Add timestep event if specified
    if args.timestep_idx is not None and args.timestep_idx < len(events):
        events[args.timestep_idx]['timestep'] = 1
        events[args.timestep_idx]['spikes'] = 0
    
    return events


def generate_corners_pattern(args):
    """Generate events in four corners of image"""
    events = []
    corners = [(0, 0), (args.img_w-1, 0), (0, args.img_h-1), (args.img_w-1, args.img_h-1)]
    
    for i, (x, y) in enumerate(corners):
        event = {
            'timestep': 0,
            'x': x,
            'y': y,
            'spikes': 1 << (i % args.in_channels)  # Different channel per corner, guaranteed spike
        }
        events.append(event)
    
    # Add additional random events to reach desired count
    while len(events) < args.num_events:
        x = random.randint(0, args.img_w - 1)
        y = random.randint(0, args.img_h - 1)
        # Skip if this coordinate already exists
        if not any(e['x'] == x and e['y'] == y for e in events):
            # Guarantee at least one spike
            guaranteed_channel = random.randint(0, args.in_channels - 1)
            spikes = 1 << guaranteed_channel
            
            # Add additional spikes with probability
            for ch in range(args.in_channels):
                if ch != guaranteed_channel and random.random() < 0.3:
                    spikes |= (1 << ch)
                    
            event = {
                'timestep': 0,
                'x': x,
                'y': y,
                'spikes': spikes
            }
            events.append(event)
    
    # Add timestep event if specified
    if args.timestep_idx is not None and args.timestep_idx < len(events):
        events[args.timestep_idx]['timestep'] = 1
        events[args.timestep_idx]['spikes'] = 0
    
    return events


def generate_grid_pattern(args):
    """Generate regular grid pattern"""
    events = []
    step_x = max(1, args.img_w // 4)
    step_y = max(1, args.img_h // 4)
    
    # Generate grid points
    for y in range(0, args.img_h, step_y):
        for x in range(0, args.img_w, step_x):
            if len(events) < args.num_events:
                # Alternate channels in checkerboard pattern (guaranteed spike)
                channel = ((x // step_x) + (y // step_y)) % args.in_channels
                event = {
                    'timestep': 0,
                    'x': x,
                    'y': y,
                    'spikes': 1 << channel
                }
                events.append(event)
    
    # Fill remaining with random events
    while len(events) < args.num_events:
        x = random.randint(0, args.img_w - 1)
        y = random.randint(0, args.img_h - 1)
        if not any(e['x'] == x and e['y'] == y for e in events):
            # Guarantee at least one spike
            guaranteed_channel = random.randint(0, args.in_channels - 1)
            spikes = 1 << guaranteed_channel
            
            # Add additional spikes with probability
            for ch in range(args.in_channels):
                if ch != guaranteed_channel and random.random() < 0.3:
                    spikes |= (1 << ch)
                    
            event = {
                'timestep': 0,
                'x': x,
                'y': y,
                'spikes': spikes
            }
            events.append(event)
    
    # Add timestep event if specified
    if args.timestep_idx is not None and args.timestep_idx < len(events):
        events[args.timestep_idx]['timestep'] = 1
        events[args.timestep_idx]['spikes'] = 0
    
    return events


def events_to_torch_tensor(events: List[Dict], args) -> Tuple[np.ndarray, Dict]:
    """Convert events to PyTorch-compatible tensor format"""
    
    # Split events into time windows
    non_timestep_events = [e for e in events if e['timestep'] == 0]
    events_per_window = len(non_timestep_events) // args.time_windows
    
    # Initialize tensor: [time_windows, channels, height, width]
    tensor_shape = (args.time_windows, args.in_channels, args.img_h, args.img_w)
    feature_maps = np.zeros(tensor_shape, dtype=np.float32)
    
    for window_idx in range(args.time_windows):
        start_idx = window_idx * events_per_window
        end_idx = start_idx + events_per_window
        if window_idx == args.time_windows - 1:  # Last window gets remaining events
            end_idx = len(non_timestep_events)
        
        window_events = non_timestep_events[start_idx:end_idx]
        
        for event in window_events:
            x, y = event['x'], event['y']
            spikes = event['spikes']
            
            # Process each channel
            for ch in range(args.in_channels):
                if spikes & (1 << ch):  # Check if channel is active
                    if args.accumulation_mode == 'binary':
                        feature_maps[window_idx, ch, y, x] = 1.0
                    elif args.accumulation_mode == 'count':
                        feature_maps[window_idx, ch, y, x] += 1.0
                    elif args.accumulation_mode == 'weighted':
                        # Weight by event order (later events have higher weight)
                        weight = (start_idx + len([e for e in window_events if e == event])) / len(non_timestep_events)
                        feature_maps[window_idx, ch, y, x] += weight
    
    # Normalize if requested
    if args.normalize_torch:
        max_val = feature_maps.max()
        if max_val > 0:
            feature_maps = feature_maps / max_val
    
    # Metadata
    metadata = {
        'shape': tensor_shape,
        'total_events': len(events),
        'non_timestep_events': len(non_timestep_events),
        'events_per_window': events_per_window,
        'accumulation_mode': args.accumulation_mode,
        'normalized': args.normalize_torch,
        'time_windows': args.time_windows,
        'img_size': [args.img_w, args.img_h],
        'channels': args.in_channels,
        'pattern': args.pattern,
        'seed': args.seed
    }
    
    return feature_maps, metadata


def print_event_samples(events, max_samples=10):
    """Print first up to 10 events as samples"""
    print(f"\nGenerated {len(events)} events")
    print("Sample events (first up to {}):\n".format(min(max_samples, len(events))))
    
    for i in range(min(max_samples, len(events))):
        event = events[i]
        spike_str = format(event['spikes'], f'0{max(1, len(bin(event["spikes"])[2:]))}b')
        print(f"  Event {i:2d}: ts={event['timestep']}, x={event['x']:2d}, y={event['y']:2d}, "
              f"spikes=0b{spike_str} (0x{event['spikes']:X})")


def save_events_to_binary_file(events, args):
    """Save events to SystemVerilog-friendly binary file"""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Calculate bit packing parameters
    event_width = 1 + 2 * args.bits_per_coord + args.in_channels
    
    filename = output_dir / f"{args.prefix}_events.mem"
    
    with open(filename, 'w') as f:
        f.write(f"// Generated event file for SNN testbench (binary format)\n")
        f.write(f"// Format: [timestep][x][y][spikes]\n")
        f.write(f"// Total bits: {event_width}\n")
        f.write(f"// Events: {len(events)}\n")
        
        # Calculate bit positions for documentation
        spikes_lsb = 0
        spikes_msb = args.in_channels - 1
        y_lsb = args.in_channels
        y_msb = args.in_channels + args.bits_per_coord - 1
        x_lsb = args.in_channels + args.bits_per_coord
        x_msb = args.in_channels + 2 * args.bits_per_coord - 1
        ts_bit = event_width - 1
        
        f.write(f"// Bit layout: timestep({ts_bit}), x({x_msb}:{x_lsb}), y({y_msb}:{y_lsb}), spikes({spikes_msb}:{spikes_lsb})\n\n")
        
        for i, event in enumerate(events):
            # Pack event into binary: [timestep][x][y][spikes]
            ts_bin = f"{event['timestep']:01b}"
            x_bin = f"{event['x']:0{args.bits_per_coord}b}"
            y_bin = f"{event['y']:0{args.bits_per_coord}b}"
            spikes_bin = f"{event['spikes']:0{args.in_channels}b}"
            
            # Concatenate to form complete binary string
            full_binary = ts_bin + x_bin + y_bin + spikes_bin
            
            # Verify total width
            assert len(full_binary) == event_width, f"Binary width mismatch: {len(full_binary)} != {event_width}"
            
            # Write binary with comment
            f.write(f"{full_binary}  // Event {i}: "
                   f"ts={event['timestep']}, x={event['x']}, y={event['y']}, "
                   f"spikes=0x{event['spikes']:X} | {ts_bin}_{x_bin}_{y_bin}_{spikes_bin}\n")
    
    print(f"\nVivado events saved to: {filename}")
    return filename


def save_torch_files(events, args):
    """Save PyTorch-compatible files"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Convert events to tensor
    feature_maps, metadata = events_to_torch_tensor(events, args)
    
    # Save numpy file
    numpy_file = output_dir / f"{args.prefix}_torch_input.npy"
    np.save(numpy_file, feature_maps)
    print(f"PyTorch tensor saved to: {numpy_file}")
    
    # Save metadata as JSON
    json_file = output_dir / f"{args.prefix}_torch_metadata.json"
    with open(json_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"PyTorch metadata saved to: {json_file}")
    
    # Save raw events as JSON for analysis
    events_json_file = output_dir / f"{args.prefix}_events.json"
    with open(events_json_file, 'w') as f:
        events_data = {
            'events': events,
            'config': {
                'in_channels': args.in_channels,
                'out_channels': args.out_channels,
                'img_size': [args.img_w, args.img_h],
                'bits_per_coord': args.bits_per_coord,
                'pattern': args.pattern,
                'seed': args.seed
            }
        }
        json.dump(events_data, f, indent=2)
    print(f"Events JSON saved to: {events_json_file}")
    
    # Print tensor statistics
    print(f"\nPyTorch tensor statistics:")
    print(f"  Shape: {feature_maps.shape}")
    print(f"  Min: {feature_maps.min():.3f}")
    print(f"  Max: {feature_maps.max():.3f}")
    print(f"  Mean: {feature_maps.mean():.3f}")
    print(f"  Non-zero elements: {np.count_nonzero(feature_maps)}/{feature_maps.size}")
    print(f"  Sparsity: {(1 - np.count_nonzero(feature_maps)/feature_maps.size)*100:.1f}%")
    
    return numpy_file, json_file, events_json_file


def main():
    """Main function"""
    args = parse_arguments()
    
    # Validate arguments
    validation_errors = validate_arguments(args)
    if validation_errors:
        print("ERROR: Invalid arguments:")
        for error in validation_errors:
            print(f"  - {error}")
        sys.exit(1)
    
    # Print parsed configuration for verification
    print("Configuration:")
    print(f"  Input channels: {args.in_channels}")
    print(f"  Output channels: {args.out_channels}")
    print(f"  Number of events: {args.num_events}")
    print(f"  Image size: {args.img_w} x {args.img_h}")
    print(f"  Bits per coordinate: {args.bits_per_coord}")
    print(f"  Timestep event index: {args.timestep_idx if args.timestep_idx is not None else 'None'}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  File prefix: {args.prefix}")
    print(f"  Pattern: {args.pattern}")
    print(f"  Random seed: {args.seed if args.seed is not None else 'None (random)'}")
    
    if args.torch_output:
        print(f"  PyTorch output: Enabled")
        print(f"  Time windows: {args.time_windows}")
        print(f"  Accumulation mode: {args.accumulation_mode}")
        print(f"  Normalize: {args.normalize_torch}")
    
    # Calculate derived values
    max_coord_val = (1 << args.bits_per_coord) - 1
    event_width = 1 + 2 * args.bits_per_coord + args.in_channels
    
    print(f"\nDerived values:")
    print(f"  Maximum coordinate value: {max_coord_val}")
    print(f"  Event width (bits): {event_width}")
    
    # Generate events
    events = generate_events(args)
    
    # Print sample events
    print_event_samples(events)
    
    # Save Vivado-compatible file
    save_events_to_binary_file(events, args)
    
    # Save PyTorch-compatible files if requested
    if args.torch_output:
        save_torch_files(events, args)
    
    print(f"\n✓ Event generation complete!")


if __name__ == "__main__":
    main()