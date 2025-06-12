#!/usr/bin/env python3
"""
SNN Event Generator for SystemVerilog Testbench
Generates event files and configuration for spiking neural network testing
"""

import argparse
import sys
import random
import math
from pathlib import Path


def parse_arguments():
    """Parse command line arguments for SNN event generation"""
    parser = argparse.ArgumentParser(
        description='Generate SNN event files and configuration for SystemVerilog testbench',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 2 4 100 8 8 7
  %(prog)s 1 2 50 16 16 8 --timestep_idx 25
  %(prog)s 3 6 200 32 32 8 --output_dir test_data --prefix my_test
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
    
    parser.add_argument('--pattern', type=str, choices=['random', 'cross', 'corners'], 
                       default='random',
                       help='Event generation pattern (default: random)')
    
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
    
    return errors


def generate_events(args):
    """Generate events based on the specified pattern"""
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    events = []
    
    if args.pattern == 'random':
        events = generate_random_events(args)
    elif args.pattern == 'cross':
        events = generate_cross_pattern(args)
    elif args.pattern == 'corners':
        events = generate_corners_pattern(args)
    
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
            # Generate spike pattern with ~30% activity per channel
            spikes = 0
            for ch in range(args.in_channels):
                if random.random() < 0.3:
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
            'spikes': 1  # Channel 0 active
        }
        events.append(event)
    
    # Vertical line (skip center to avoid duplicate)
    for y in range(max(0, center_y - 2), min(args.img_h, center_y + 3)):
        if y != center_y:
            event = {
                'timestep': 0,
                'x': center_x,
                'y': y,
                'spikes': 1  # Channel 0 active
            }
            events.append(event)
    
    # Add additional random events to reach desired count
    while len(events) < args.num_events:
        x = random.randint(0, args.img_w - 1)
        y = random.randint(0, args.img_h - 1)
        # Skip if this coordinate already exists
        if not any(e['x'] == x and e['y'] == y for e in events):
            event = {
                'timestep': 0,
                'x': x,
                'y': y,
                'spikes': random.randint(1, (1 << args.in_channels) - 1)
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
            'spikes': 1 << (i % args.in_channels)  # Different channel per corner
        }
        events.append(event)
    
    # Add additional random events to reach desired count
    while len(events) < args.num_events:
        x = random.randint(0, args.img_w - 1)
        y = random.randint(0, args.img_h - 1)
        # Skip if this coordinate already exists
        if not any(e['x'] == x and e['y'] == y for e in events):
            event = {
                'timestep': 0,
                'x': x,
                'y': y,
                'spikes': random.randint(1, (1 << args.in_channels) - 1)
            }
            events.append(event)
    
    # Add timestep event if specified
    if args.timestep_idx is not None and args.timestep_idx < len(events):
        events[args.timestep_idx]['timestep'] = 1
        events[args.timestep_idx]['spikes'] = 0
    
    return events


def print_event_samples(events, max_samples=10):
    """Print first up to 10 events as samples"""
    print(f"\nGenerated {len(events)} events")
    print("Sample events (first up to {}):\n".format(min(max_samples, len(events))))
    
    for i in range(min(max_samples, len(events))):
        event = events[i]
        spike_str = format(event['spikes'], f'0{max(1, len(bin(event["spikes"])[2:]))}b')
        print(f"  Event {i:2d}: ts={event['timestep']}, x={event['x']:2d}, y={event['y']:2d}, "
              f"spikes=0b{spike_str} (0x{event['spikes']:X})")


def save_events_to_hex_file(events, args):
    """Save events to SystemVerilog-friendly hex file"""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Calculate bit packing parameters
    event_width = 1 + 2 * args.bits_per_coord + args.in_channels
    hex_width = (event_width + 3) // 4  # Round up to nearest hex digit
    
    filename = output_dir / f"{args.prefix}_events.hex"
    
    with open(filename, 'w') as f:
        f.write(f"// Generated event file for SNN testbench\n")
        f.write(f"// Format: [timestep][x][y][spikes]\n")
        f.write(f"// Total bits: {event_width}, Hex width: {hex_width}\n")
        f.write(f"// Events: {len(events)}\n")
        f.write(f"// Bit layout: timestep({args.bits_per_coord + args.in_channels + 1}), ")
        f.write(f"x({args.bits_per_coord + args.in_channels}:{args.in_channels + 1}), ")
        f.write(f"y({args.in_channels}:1), spikes({args.in_channels - 1}:0)\n\n")
        
        for i, event in enumerate(events):
            # Pack event into binary: [timestep][x][y][spikes]
            packed_value = 0
            packed_value |= (event['timestep'] & 1) << (event_width - 1)
            packed_value |= (event['x'] & ((1 << args.bits_per_coord) - 1)) << (args.bits_per_coord + args.in_channels)
            packed_value |= (event['y'] & ((1 << args.bits_per_coord) - 1)) << args.in_channels
            packed_value |= (event['spikes'] & ((1 << args.in_channels) - 1))
            
            # Write as hex with comment
            f.write(f"{packed_value:0{hex_width}X}  // Event {i}: "
                   f"ts={event['timestep']}, x={event['x']}, y={event['y']}, "
                   f"spikes=0x{event['spikes']:X}\n")
    
    print(f"\nEvents saved to: {filename}")
    return filename


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
    print("Parsed configuration:")
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
    
    # Calculate derived values
    max_coord_val = (1 << args.bits_per_coord) - 1
    event_width = 1 + 2 * args.bits_per_coord + args.in_channels
    
    print(f"\nDerived values:")
    print(f"  Maximum coordinate value: {max_coord_val}")
    print(f"  Event width (bits): {event_width}")
    print(f"  Event width (hex digits): {(event_width + 3) // 4}")
    
    # Generate events
    events = generate_events(args)
    
    # Print sample events
    print_event_samples(events)
    
    # Save to file
    save_events_to_hex_file(events, args)


if __name__ == "__main__":
    main()