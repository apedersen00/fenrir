#!/usr/bin/env python3
"""
Correct Accumulating Convolution Verification
Accumulates spike events and applies conv2d with transposed kernel weights
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

# Configuration
IMG_WIDTH = 32
IMG_HEIGHT = 32
CHANNELS = 6

def load_systemverilog_weights():
    weights = np.array([
        [  1, 1,  0,  0, -1,  -1],   # Position 0
        [  2, 1,  0,  0, -1,  -2],   # Position 1
        [  1, 1,  0,  0, -1,  -1],   # Position 2
        [  0, 0,  2,  1, -1,  -2],   # Position 3
        [  0, 0,  0,  0,  0,  0],   # Position 4
        [  0, 0, -2, -1,  1, 2],   # Position 5
        [ -1,-1,  0,  0,  1,  1],   # Position 6
        [ -2,-1,  0,  0,  1,  2],   # Position 7
        [ -1,-1,  0,  0,  1,  1]    # Position 8
    ], dtype=np.float32)
    return weights

def create_transposed_conv2d_layer(sv_weights):
    """
    Create conv2d layer with transposed kernel weights
    
    SystemVerilog applies kernels in transposed orientation:
    
    Original SystemVerilog:     Transposed for PyTorch:
    0 1 2                      0 3 6
    3 4 5          →           1 4 7  
    6 7 8                      2 5 8
    
    This creates vertical patterns instead of horizontal ones.
    """
    conv = nn.Conv2d(1, CHANNELS, 3, stride=1, padding=1, bias=False, padding_mode='zeros')
    
    weight_tensor = torch.zeros(CHANNELS, 1, 3, 3)
    
    # Transpose mapping: PyTorch[row,col] ← SystemVerilog[sv_pos]
    transpose_mapping = [
        (0, 0, 0), (0, 1, 3), (0, 2, 6),  # Top row ← Left column
        (1, 0, 1), (1, 1, 4), (1, 2, 7),  # Mid row ← Mid column  
        (2, 0, 2), (2, 1, 5), (2, 2, 8)   # Bottom row ← Right column
    ]
    
    for ch in range(CHANNELS):
        for pt_row, pt_col, sv_pos in transpose_mapping:
            weight_tensor[ch, 0, pt_row, pt_col] = float(sv_weights[sv_pos, ch])
    
    conv.weight.data = weight_tensor
    return conv

def load_events(filename):
    """Load events from file"""
    events = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('//'):
                parts = line.split(',')
                if len(parts) >= 2:
                    x, y = int(parts[0]), int(parts[1])
                    events.append((x, y))
    return events

def generate_accumulating_conv_states(events, conv_layer):
    """
    Generate memory states by accumulating spikes and applying conv2d after each event
    """
    print(f"🔄 Processing {len(events)} events with accumulating convolution...")
    
    # Start with empty accumulated image
    accumulated_image = torch.zeros(1, 1, IMG_HEIGHT, IMG_WIDTH, dtype=torch.float32)
    memory_states = []
    
    with torch.no_grad():
        for event_idx, (x, y) in enumerate(events):
            # Add this event to the accumulated image
            if 0 <= x < IMG_WIDTH and 0 <= y < IMG_HEIGHT:
                accumulated_image[0, 0, y, x] += 1.0
                
                # Apply convolution to current accumulated image
                conv_output = conv_layer(accumulated_image)
                
                # Apply saturation (like SystemVerilog: 6-bit signed)
                conv_output = torch.clamp(conv_output, min=-32, max=31)
                
                # Convert to numpy format [height, width, channels]
                memory_state = conv_output[0].permute(1, 2, 0).numpy().astype(np.int8)
                memory_states.append(memory_state)
                
                # Debug info
                total_spikes = torch.sum(accumulated_image).item()
                nonzero_memory = np.count_nonzero(memory_state)
                print(f"  Event {event_idx}: spike at ({x},{y}) | "
                      f"Total spikes: {total_spikes:.0f} | "
                      f"Non-zero memory: {nonzero_memory}")
            else:
                print(f"  Event {event_idx}: spike at ({x},{y}) - OUT OF BOUNDS, skipping")
    
    print(f"✅ Generated {len(memory_states)} memory states")
    return memory_states

def load_systemverilog_memory_states(dump_directory):
    """Load memory states from SystemVerilog CSV files"""
    dump_dir = Path(dump_directory)
    print(f"📁 Loading SystemVerilog memory dumps from: {dump_dir}")
    
    if not dump_dir.exists():
        print(f"❌ Directory not found: {dump_dir}")
        return None
    
    # Find memory dump files
    dump_files = list(dump_dir.glob("memory_dump_event_*.csv"))
    if not dump_files:
        print("❌ No memory_dump_event_*.csv files found")
        return None
    
    dump_files.sort(key=lambda f: int(f.stem.split('_')[-1]))
    print(f"📂 Found {len(dump_files)} dump files")
    
    memory_states = {}
    
    for dump_file in dump_files:
        try:
            print(f"  📄 Loading {dump_file.name}...")
            df = pd.read_csv(dump_file)
            
            # Create memory array [height, width, channels]
            memory = np.zeros((IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.int8)
            
            # Check CSV format
            expected_cols = ['x', 'y'] + [f'ch{i}' for i in range(CHANNELS)]
            if not all(col in df.columns for col in expected_cols):
                print(f"    ⚠️  Missing expected columns in {dump_file.name}")
                print(f"        Expected: {expected_cols}")
                print(f"        Found: {list(df.columns)}")
                continue
            
            # Load data from CSV
            for _, row in df.iterrows():
                try:
                    x, y = int(row['x']), int(row['y'])
                    if 0 <= x < IMG_WIDTH and 0 <= y < IMG_HEIGHT:
                        for ch in range(CHANNELS):
                            memory[y, x, ch] = int(row[f'ch{ch}'])
                except (ValueError, KeyError):
                    continue
            
            # Map CSV file to event index
            # memory_dump_event_1.csv = after event 0, so event_num-1
            event_num = int(dump_file.stem.split('_')[-1])
            if event_num > 0:
                memory_states[event_num - 1] = memory
            
            nonzero_count = np.count_nonzero(memory)
            print(f"    ✅ Event {event_num-1}: {len(df)} CSV entries, {nonzero_count} non-zero values")
            
        except Exception as e:
            print(f"    ❌ Error loading {dump_file.name}: {e}")
    
    print(f"✅ Loaded {len(memory_states)} SystemVerilog memory states")
    return memory_states

def compare_memory_states(pytorch_states, sv_states, events):
    """Compare PyTorch and SystemVerilog memory states"""
    print(f"\n🔍 Comparing PyTorch vs SystemVerilog Memory States")
    print("=" * 55)
    
    total_errors = 0
    perfect_matches = 0
    
    for event_idx in range(len(pytorch_states)):
        if event_idx not in sv_states:
            print(f"❌ Event {event_idx}: No SystemVerilog data")
            continue
        
        pytorch_memory = pytorch_states[event_idx]
        sv_memory = sv_states[event_idx]
        
        # Compare memories
        differences = (pytorch_memory != sv_memory)
        error_count = np.sum(differences)
        
        # Calculate accuracy
        mask = (pytorch_memory != 0) | (sv_memory != 0)
        total_compared = np.sum(mask)
        
        if total_compared > 0:
            accuracy = ((total_compared - error_count) / total_compared) * 100
        else:
            accuracy = 100.0
        
        total_errors += error_count
        
        if error_count == 0:
            perfect_matches += 1
            status = "✅"
        else:
            status = "❌"
        
        event_x, event_y = events[event_idx] if event_idx < len(events) else ("?", "?")
        print(f"  {status} Event {event_idx} ({event_x},{event_y}): "
              f"{error_count}/{total_compared} errors ({accuracy:.1f}% accurate)")
        
        # Show first few errors for failed events
        if error_count > 0 and error_count <= 10:
            print(f"      🔍 First few differences:")
            diff_coords = np.where(differences)
            for i in range(min(3, len(diff_coords[0]))):
                y, x, ch = diff_coords[0][i], diff_coords[1][i], diff_coords[2][i]
                pt_val = pytorch_memory[y, x, ch]
                sv_val = sv_memory[y, x, ch]
                print(f"         ({x},{y}) Ch{ch}: PyTorch={pt_val}, SV={sv_val}")
    
    # Overall summary
    events_compared = len([e for e in range(len(pytorch_states)) if e in sv_states])
    
    print(f"\n📊 VERIFICATION RESULTS:")
    print(f"   Events compared: {events_compared}")
    print(f"   Perfect matches: {perfect_matches}/{events_compared}")
    print(f"   Total errors: {total_errors}")
    
    if total_errors == 0:
        print(f"\n🎉 PERFECT MATCH!")
        print(f"   ✅ Transposed kernel weights work correctly!")
        print(f"   ✅ Accumulating convolution matches SystemVerilog!")
        return True
    else:
        accuracy_overall = ((events_compared * 100) - total_errors) / (events_compared * 100) * 100 if events_compared > 0 else 0
        print(f"\n⚠️  Verification failed:")
        print(f"   ❌ Overall accuracy: {accuracy_overall:.1f}%")
        print(f"   🔧 Need to check kernel weight mapping or convolution logic")
        return False

def show_kernel_weights_comparison():
    """Show SystemVerilog vs transposed PyTorch kernel weights"""
    print("🔍 Kernel Weights Comparison")
    print("=" * 40)
    
    sv_weights = load_systemverilog_weights()
    conv = create_transposed_conv2d_layer(sv_weights)
    
    print("📊 SystemVerilog kernel weights (positions 0-8):")
    print("Position layout:")
    print("  0 1 2")
    print("  3 4 5")
    print("  6 7 8")
    print()
    
    for ch in range(CHANNELS):
        print(f"Channel {ch}:")
        for pos in range(9):
            row, col = pos // 3, pos % 3
            print(f"  Pos{pos} ({row},{col}): {sv_weights[pos, ch]:3.0f}", end="")
            if pos % 3 == 2:
                print()
        print()
    
    print("🔄 PyTorch transposed kernel weights:")
    print("PyTorch layout:")
    print("  [0,0] [0,1] [0,2]")
    print("  [1,0] [1,1] [1,2]")
    print("  [2,0] [2,1] [2,2]")
    print()
    
    for ch in range(CHANNELS):
        print(f"Channel {ch}:")
        for row in range(3):
            for col in range(3):
                val = conv.weight.data[ch, 0, row, col].item()
                print(f"  [{row},{col}]: {val:3.0f}", end="")
            print()
        print()
        
    print("🔄 Transpose mapping explanation:")
    print("  SystemVerilog pos 0,1,2 → PyTorch [0,0], [1,0], [2,0] (left column)")
    print("  SystemVerilog pos 3,4,5 → PyTorch [0,1], [1,1], [2,1] (mid column)")  
    print("  SystemVerilog pos 6,7,8 → PyTorch [0,2], [1,2], [2,2] (right column)")
    print("  This creates vertical patterns instead of horizontal ones.")

def test_single_event(spike_x, spike_y, dump_directory):
    """Test a single event and show detailed comparison"""
    print(f"\n🎯 Testing Single Event: spike at ({spike_x},{spike_y})")
    print("=" * 50)
    
    # Create PyTorch result
    sv_weights = load_systemverilog_weights()
    conv = create_transposed_conv2d_layer(sv_weights)
    
    # Single event on empty image
    image = torch.zeros(1, 1, IMG_HEIGHT, IMG_WIDTH, dtype=torch.float32)
    image[0, 0, spike_y, spike_x] = 1.0
    
    with torch.no_grad():
        pytorch_output = conv(image)
        pytorch_output = torch.clamp(pytorch_output, min=-32, max=31)
        pytorch_memory = pytorch_output[0].permute(1, 2, 0).numpy().astype(np.int8)
    
    print(f"📊 PyTorch result: {np.count_nonzero(pytorch_memory)} non-zero values")
    
    # Load corresponding SystemVerilog result
    sv_states = load_systemverilog_memory_states(dump_directory)
    if sv_states and 0 in sv_states:
        sv_memory = sv_states[0]
        
        differences = (pytorch_memory != sv_memory)
        error_count = np.sum(differences)
        
        print(f"📊 SystemVerilog result: {np.count_nonzero(sv_memory)} non-zero values")
        print(f"🔍 Comparison: {error_count} differences")
        
        if error_count == 0:
            print("✅ Perfect match!")
        else:
            print("❌ Differences found:")
            
            # Show 5x5 region around spike
            start_y = max(0, spike_y - 2)
            end_y = min(IMG_HEIGHT, spike_y + 3)
            start_x = max(0, spike_x - 2)
            end_x = min(IMG_WIDTH, spike_x + 3)
            
            for ch in range(min(2, CHANNELS)):  # Show first 2 channels
                print(f"\nChannel {ch} - 5x5 region around ({spike_x},{spike_y}):")
                
                print("PyTorch:")
                for y in range(start_y, end_y):
                    print(f"  {y:2d}: ", end="")
                    for x in range(start_x, end_x):
                        val = pytorch_memory[y, x, ch]
                        print(f"{val:4d}" if val != 0 else "   .", end="")
                    print()
                
                print("SystemVerilog:")
                for y in range(start_y, end_y):
                    print(f"  {y:2d}: ", end="")
                    for x in range(start_x, end_x):
                        val = sv_memory[y, x, ch]
                        print(f"{val:4d}" if val != 0 else "   .", end="")
                    print()

def main():
    parser = argparse.ArgumentParser(description='Accumulating convolution verification with transposed kernel weights')
    parser.add_argument('dump_directory', nargs='?', help='Directory with SystemVerilog dumps')
    parser.add_argument('-e', '--events', help='Events file')
    parser.add_argument('--show-weights', action='store_true', help='Show kernel weight comparison')
    parser.add_argument('--test-single', nargs=2, type=int, help='Test single event (x y)')
    
    args = parser.parse_args()
    
    print("🔄 Accumulating Convolution Verification")
    print("=" * 45)
    print("Accumulates spikes and applies conv2d with transposed kernel weights")
    
    if args.show_weights:
        show_kernel_weights_comparison()
        return 0
    
    if args.test_single and args.dump_directory:
        spike_x, spike_y = args.test_single
        test_single_event(spike_x, spike_y, args.dump_directory)
        return 0
    
    if not args.dump_directory or not args.events:
        print("\n💡 Usage examples:")
        print("  python correct_accumulating_conv.py --show-weights")
        print("  python correct_accumulating_conv.py mem_dumps --test-single 5 5")
        print("  python correct_accumulating_conv.py mem_dumps -e test_events.txt")
        return 1
    
    # Load events
    events = load_events(args.events)
    print(f"📝 Loaded {len(events)} events")
    
    # Create conv layer with transposed weights
    sv_weights = load_systemverilog_weights()
    conv_layer = create_transposed_conv2d_layer(sv_weights)
    print(f"⚙️  Created conv2d layer with transposed kernel weights")
    
    # Generate PyTorch memory states
    pytorch_states = generate_accumulating_conv_states(events, conv_layer)
    
    # Load SystemVerilog memory states
    sv_states = load_systemverilog_memory_states(args.dump_directory)
    if sv_states is None:
        return 1
    
    # Compare results
    success = compare_memory_states(pytorch_states, sv_states, events)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())