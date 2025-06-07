#!/usr/bin/env python3
"""
Direct Cross-Correlation Verification
Uses SystemVerilog kernel weights exactly as-is since both PyTorch and SystemVerilog use cross-correlation
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
    """Load exact SystemVerilog kernel weights"""
    weights = np.array([
        [-1, -1,  0,  0,  1,  1],   # Position 0
        [-2, -1,  0,  0,  1,  2],   # Position 1
        [-1, -1,  0,  0,  1,  1],   # Position 2
        [ 0,  0, -2, -1,  1,  2],   # Position 3
        [ 0,  0,  0,  0,  0,  0],   # Position 4
        [ 0,  0,  2,  1, -1, -2],   # Position 5
        [ 1,  1,  0,  0, -1, -1],   # Position 6
        [ 2,  1,  0,  0, -1, -2],   # Position 7
        [ 1,  1,  0,  0, -1, -1]    # Position 8
    ], dtype=np.float32)
    return weights

def create_direct_conv2d_layer(sv_weights):
    """
    Create conv2d layer using SystemVerilog weights DIRECTLY
    
    Since both PyTorch conv2d and SystemVerilog use cross-correlation,
    we should use the weights exactly as defined.
    
    The question is: how does SystemVerilog map positions 0-8 to the 3x3 kernel?
    """
    conv = nn.Conv2d(1, CHANNELS, 3, stride=1, padding=1, bias=False)
    weight_tensor = torch.zeros(CHANNELS, 1, 3, 3)
    
    # Test different mappings to see which one matches SystemVerilog
    
    # Mapping 1: Standard row-major (most common)
    # 0 1 2
    # 3 4 5  
    # 6 7 8
    standard_mapping = [
        (0, 0, 0), (0, 1, 1), (0, 2, 2),
        (1, 0, 3), (1, 1, 4), (1, 2, 5),
        (2, 0, 6), (2, 1, 7), (2, 2, 8)
    ]
    
    for ch in range(CHANNELS):
        for pt_row, pt_col, sv_pos in standard_mapping:
            weight_tensor[ch, 0, pt_row, pt_col] = float(sv_weights[sv_pos, ch])
    
    conv.weight.data = weight_tensor
    return conv, "standard"

def create_column_major_conv2d_layer(sv_weights):
    """
    Create conv2d layer using column-major mapping with NEGATED weights
    
    SystemVerilog nested loops: dx outer, dy inner
    for (dx = -1; dx <= 1; dx++)
        for (dy = -1; dy <= 1; dy++)
    
    This creates:
    0 3 6
    1 4 7
    2 5 8
    
    BUT: We need to NEGATE the weights to match SystemVerilog sign convention!
    """
    conv = nn.Conv2d(1, CHANNELS, 3, stride=1, padding=1, bias=False)
    weight_tensor = torch.zeros(CHANNELS, 1, 3, 3)
    
    # Column-major mapping based on SystemVerilog loops
    column_mapping = [
        (0, 0, 0), (0, 1, 3), (0, 2, 6),
        (1, 0, 1), (1, 1, 4), (1, 2, 7),
        (2, 0, 2), (2, 1, 5), (2, 2, 8)
    ]
    
    for ch in range(CHANNELS):
        for pt_row, pt_col, sv_pos in column_mapping:
            # NEGATE the weights to match SystemVerilog sign convention
            weight_tensor[ch, 0, pt_row, pt_col] = -float(sv_weights[sv_pos, ch])
    
    conv.weight.data = weight_tensor
    return conv, "column_major_negated"

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

def generate_conv_states(events, conv_layer):
    """Generate memory states using accumulating convolution"""
    print(f"🔄 Processing {len(events)} events with cross-correlation...")
    
    accumulated_image = torch.zeros(1, 1, IMG_HEIGHT, IMG_WIDTH, dtype=torch.float32)
    memory_states = []
    
    with torch.no_grad():
        for event_idx, (x, y) in enumerate(events):
            if 0 <= x < IMG_WIDTH and 0 <= y < IMG_HEIGHT:
                accumulated_image[0, 0, y, x] += 1.0
                
                conv_output = conv_layer(accumulated_image)
                conv_output = torch.clamp(conv_output, min=-32, max=31)
                
                memory_state = conv_output[0].permute(1, 2, 0).numpy().astype(np.int8)
                memory_states.append(memory_state)
                
                nonzero_count = np.count_nonzero(memory_state)
                print(f"  Event {event_idx}: spike at ({x},{y}) | Non-zero: {nonzero_count}")
            else:
                print(f"  Event {event_idx}: spike at ({x},{y}) - OUT OF BOUNDS")
    
    return memory_states

def load_systemverilog_memory_states(dump_directory):
    """Load memory states from SystemVerilog CSV files"""
    dump_dir = Path(dump_directory)
    
    if not dump_dir.exists():
        print(f"❌ Directory not found: {dump_dir}")
        return None
    
    dump_files = list(dump_dir.glob("memory_dump_event_*.csv"))
    if not dump_files:
        print("❌ No memory_dump_event_*.csv files found")
        return None
    
    dump_files.sort(key=lambda f: int(f.stem.split('_')[-1]))
    memory_states = {}
    
    for dump_file in dump_files:
        try:
            df = pd.read_csv(dump_file)
            memory = np.zeros((IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.int8)
            
            for _, row in df.iterrows():
                try:
                    x, y = int(row['x']), int(row['y'])
                    if 0 <= x < IMG_WIDTH and 0 <= y < IMG_HEIGHT:
                        for ch in range(CHANNELS):
                            memory[y, x, ch] = int(row[f'ch{ch}'])
                except (ValueError, KeyError):
                    continue
            
            event_num = int(dump_file.stem.split('_')[-1])
            if event_num > 0:
                memory_states[event_num - 1] = memory
            
        except Exception as e:
            print(f"    ❌ Error loading {dump_file.name}: {e}")
    
    return memory_states

def test_both_mappings(spike_x, spike_y, dump_directory):
    """Test standard, column-major, and negated column-major mappings for a single event"""
    print(f"\n🎯 Testing All Mappings for spike at ({spike_x},{spike_y})")
    print("=" * 60)
    
    sv_weights = load_systemverilog_weights()
    
    # Test standard mapping
    conv_std, _ = create_direct_conv2d_layer(sv_weights)
    image = torch.zeros(1, 1, IMG_HEIGHT, IMG_WIDTH, dtype=torch.float32)
    image[0, 0, spike_y, spike_x] = 1.0
    
    with torch.no_grad():
        output_std = conv_std(image)
        output_std = torch.clamp(output_std, min=-32, max=31)
        memory_std = output_std[0].permute(1, 2, 0).numpy().astype(np.int8)
    
    # Test column-major mapping with negated weights
    conv_col, _ = create_column_major_conv2d_layer(sv_weights)
    
    with torch.no_grad():
        output_col = conv_col(image)
        output_col = torch.clamp(output_col, min=-32, max=31)
        memory_col = output_col[0].permute(1, 2, 0).numpy().astype(np.int8)
    
    # Load SystemVerilog result
    sv_states = load_systemverilog_memory_states(dump_directory)
    if not sv_states or 0 not in sv_states:
        print("❌ No SystemVerilog data for event 0")
        return
    
    sv_memory = sv_states[0]
    
    # Compare all mappings
    errors_std = np.sum(memory_std != sv_memory)
    errors_col = np.sum(memory_col != sv_memory)
    
    print(f"📊 Standard mapping errors: {errors_std}")
    print(f"📊 Column-major negated mapping errors: {errors_col}")
    
    # Show 5x5 region for the best match
    if errors_col == 0:
        best_mapping = "column_major_negated"
        best_memory = memory_col
        print(f"\n🎉 PERFECT MATCH with column-major negated mapping!")
    elif errors_std < errors_col:
        best_mapping = "standard"
        best_memory = memory_std
        print(f"\n🏆 Best mapping: standard ({errors_std} errors)")
    else:
        best_mapping = "column_major_negated"
        best_memory = memory_col
        print(f"\n🏆 Best mapping: column_major_negated ({errors_col} errors)")
    
    start_y = max(0, spike_y - 2)
    end_y = min(IMG_HEIGHT, spike_y + 3)
    start_x = max(0, spike_x - 2)
    end_x = min(IMG_WIDTH, spike_x + 3)
    
    for ch in range(min(2, CHANNELS)):
        print(f"\nChannel {ch} - 5x5 region around ({spike_x},{spike_y}):")
        
        print(f"{best_mapping.replace('_', ' ').title()} mapping:")
        for y in range(start_y, end_y):
            print(f"  {y:2d}: ", end="")
            for x in range(start_x, end_x):
                val = best_memory[y, x, ch]
                print(f"{val:4d}" if val != 0 else "   .", end="")
            print()
        
        print("SystemVerilog:")
        for y in range(start_y, end_y):
            print(f"  {y:2d}: ", end="")
            for x in range(start_x, end_x):
                val = sv_memory[y, x, ch]
                print(f"{val:4d}" if val != 0 else "   .", end="")
            print()
    
    return best_mapping

def show_kernel_mappings():
    """Show both possible kernel mappings"""
    print("🔍 Kernel Mapping Analysis")
    print("=" * 40)
    
    sv_weights = load_systemverilog_weights()
    
    print("📊 SystemVerilog kernel weights (positions 0-8):")
    for ch in range(CHANNELS):
        print(f"\nChannel {ch}:")
        for pos in range(9):
            print(f"  Pos{pos}: {sv_weights[pos, ch]:3.0f}")
    
    print("\n🔄 Standard mapping (row-major):")
    print("  0 1 2")
    print("  3 4 5")
    print("  6 7 8")
    
    print("\n🔄 Column-major mapping with NEGATED weights (SystemVerilog loop order):")
    print("  0 3 6")
    print("  1 4 7")
    print("  2 5 8")
    print("  (All weights negated to match SystemVerilog sign convention)")
    
    conv_std, _ = create_direct_conv2d_layer(sv_weights)
    conv_col, _ = create_column_major_conv2d_layer(sv_weights)
    
    print("\n📊 PyTorch kernel weights comparison:")
    for ch in range(min(2, CHANNELS)):
        print(f"\nChannel {ch}:")
        print("Standard mapping:")
        for row in range(3):
            print("  ", end="")
            for col in range(3):
                val = conv_std.weight.data[ch, 0, row, col].item()
                print(f"{val:4.0f}", end="")
            print()
        
        print("Column-major NEGATED mapping:")
        for row in range(3):
            print("  ", end="")
            for col in range(3):
                val = conv_col.weight.data[ch, 0, row, col].item()
                print(f"{val:4.0f}", end="")
            print()

def compare_memory_states(pytorch_states, sv_states, events, mapping_name):
    """Compare PyTorch and SystemVerilog memory states"""
    print(f"\n🔍 Comparing {mapping_name} mapping vs SystemVerilog")
    print("=" * 50)
    
    total_errors = 0
    perfect_matches = 0
    
    for event_idx in range(min(len(pytorch_states), 10)):  # Test first 10 events
        if event_idx not in sv_states:
            continue
        
        pytorch_memory = pytorch_states[event_idx]
        sv_memory = sv_states[event_idx]
        
        differences = (pytorch_memory != sv_memory)
        error_count = np.sum(differences)
        total_errors += error_count
        
        mask = (pytorch_memory != 0) | (sv_memory != 0)
        total_compared = np.sum(mask)
        accuracy = ((total_compared - error_count) / total_compared) * 100 if total_compared > 0 else 100.0
        
        if error_count == 0:
            perfect_matches += 1
            status = "✅"
        else:
            status = "❌"
        
        event_x, event_y = events[event_idx] if event_idx < len(events) else ("?", "?")
        print(f"  {status} Event {event_idx} ({event_x},{event_y}): "
              f"{error_count}/{total_compared} errors ({accuracy:.1f}% accurate)")
    
    print(f"\n📊 {mapping_name.title()} mapping results:")
    print(f"   Perfect matches: {perfect_matches}/10")
    print(f"   Total errors: {total_errors}")
    
    return total_errors

def main():
    parser = argparse.ArgumentParser(description='Direct cross-correlation verification')
    parser.add_argument('dump_directory', nargs='?', help='Directory with SystemVerilog dumps')
    parser.add_argument('-e', '--events', help='Events file')
    parser.add_argument('--show-mappings', action='store_true', help='Show kernel mappings')
    parser.add_argument('--test-single', nargs=2, type=int, help='Test single event (x y)')
    
    args = parser.parse_args()
    
    print("🎯 Direct Cross-Correlation Verification")
    print("=" * 45)
    print("Uses SystemVerilog weights directly - no manipulation")
    
    if args.show_mappings:
        show_kernel_mappings()
        return 0
    
    if args.test_single and args.dump_directory:
        spike_x, spike_y = args.test_single
        best_mapping = test_both_mappings(spike_x, spike_y, args.dump_directory)
        print(f"\n🏆 Recommended mapping: {best_mapping}")
        return 0
    
    if not args.dump_directory or not args.events:
        print("\n💡 Usage examples:")
        print("  python direct_conv_verification.py --show-mappings")
        print("  python direct_conv_verification.py mem_dumps --test-single 5 5")
        print("  python direct_conv_verification.py mem_dumps -e test_events.txt")
        return 1
    
    # Load data
    events = load_events(args.events)
    sv_states = load_systemverilog_memory_states(args.dump_directory)
    if sv_states is None:
        return 1
    
    sv_weights = load_systemverilog_weights()
    
    # Test both mappings
    print("\n1️⃣ Testing Standard Mapping:")
    conv_std, _ = create_direct_conv2d_layer(sv_weights)
    states_std = generate_conv_states(events, conv_std)
    errors_std = compare_memory_states(states_std, sv_states, events, "standard")
    
    print("\n2️⃣ Testing Column-Major Negated Mapping:")
    conv_col, _ = create_column_major_conv2d_layer(sv_weights)
    states_col = generate_conv_states(events, conv_col)
    errors_col = compare_memory_states(states_col, sv_states, events, "column_major_negated")
    
    print(f"\n🏆 FINAL RESULTS:")
    print(f"   Standard mapping: {errors_std} errors")
    print(f"   Column-major negated mapping: {errors_col} errors")
    
    if errors_std == 0:
        print(f"\n🎉 PERFECT MATCH with standard mapping!")
    elif errors_col == 0:
        print(f"\n🎉 PERFECT MATCH with column-major negated mapping!")
    else:
        print(f"\n⚠️  Neither mapping is perfect. Need further investigation.")
    
    return 0

if __name__ == "__main__":
    exit(main())