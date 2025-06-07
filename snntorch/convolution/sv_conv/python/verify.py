#!/usr/bin/env python3
"""
Sequential Event-by-Event Conv2d Verification
Builds memory representation that matches SystemVerilog event processing
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import re

# Configuration
IMG_WIDTH = 32
IMG_HEIGHT = 32
CHANNELS = 6
KERNEL_SIZE = 3

def load_systemverilog_weights():
    """Load exact SystemVerilog kernel weights"""
    weights = np.array([
        [-1, -1,  0,  0,  1,  1],   # Position 0 (top-left)
        [-2, -1,  0,  0,  1,  2],   # Position 1 (top-center)  
        [-1, -1,  0,  0,  1,  1],   # Position 2 (top-right)
        [ 0,  0, -2, -1,  1,  2],   # Position 3 (mid-left)
        [ 0,  0,  0,  0,  0,  0],   # Position 4 (center)
        [ 0,  0,  2,  1, -1, -2],   # Position 5 (mid-right)
        [ 1,  1,  0,  0, -1, -1],   # Position 6 (bottom-left)
        [ 2,  1,  0,  0, -1, -2],   # Position 7 (bottom-center)
        [ 1,  1,  0,  0, -1, -1]    # Position 8 (bottom-right)
    ], dtype=np.float32)
    
    return weights

def create_conv2d_layer(sv_weights):
    """Create conv2d layer with SystemVerilog weights"""
    conv = nn.Conv2d(1, CHANNELS, KERNEL_SIZE, stride=1, padding=1, bias=False)
    
    # Convert to PyTorch format with kernel flipping
    # nn.Conv2d performs convolution (flips kernel), but SystemVerilog does cross-correlation
    weight_tensor = torch.zeros(CHANNELS, 1, KERNEL_SIZE, KERNEL_SIZE)
    for ch in range(CHANNELS):
        for pos in range(9):
            # SystemVerilog position mapping
            sv_row = pos // 3
            sv_col = pos % 3
            
            # Flip kernel for true convolution: (row,col) -> (2-row, 2-col)
            pytorch_row = 2 - sv_row
            pytorch_col = 2 - sv_col
            
            weight_tensor[ch, 0, pytorch_row, pytorch_col] = float(sv_weights[pos, ch])
    
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

def generate_sequential_memory_states(events, conv_layer):
    """
    Generate memory states sequentially, just like SystemVerilog:
    - Add each event to accumulated image
    - Apply conv2d 
    - Store resulting memory state
    """
    print(f"🔄 Processing {len(events)} events sequentially...")
    
    # Start with empty accumulated image
    accumulated_image = torch.zeros(1, 1, IMG_HEIGHT, IMG_WIDTH, dtype=torch.float32)
    memory_states = []
    
    with torch.no_grad():
        for event_idx, (x, y) in enumerate(events):
            # Add this event to the accumulated image
            if 0 <= x < IMG_WIDTH and 0 <= y < IMG_HEIGHT:
                accumulated_image[0, 0, y, x] += 1.0
                
                # Apply convolution to current accumulated image
                memory_state = conv_layer(accumulated_image)
                
                # Apply saturation (like SystemVerilog)
                memory_state = torch.clamp(memory_state, min=-32, max=31)
                
                # Store this memory state
                memory_states.append(memory_state.clone())
                
                # Debug info
                total_spikes = torch.sum(accumulated_image).item()
                nonzero_memory = torch.count_nonzero(memory_state).item()
                print(f"  Event {event_idx}: spike at ({x},{y}) | "
                      f"Total spikes: {total_spikes:.0f} | "
                      f"Non-zero memory: {nonzero_memory}")
            else:
                print(f"  Event {event_idx}: spike at ({x},{y}) - OUT OF BOUNDS, skipping")
    
    print(f"✅ Generated {len(memory_states)} memory states")
    return memory_states

def load_systemverilog_memory_states(dump_directory):
    """Load memory states from directory of SystemVerilog memory dump files"""
    dump_dir = Path(dump_directory)
    print(f"📁 Loading SystemVerilog memory dumps from directory: {dump_dir}")
    
    if not dump_dir.exists():
        print(f"❌ Directory not found: {dump_dir}")
        return None
    
    # Find memory dump files - try different naming patterns
    dump_files = []
    
    # Pattern 1: memory_dump_event_N.csv
    pattern1_files = list(dump_dir.glob("memory_dump_event_*.csv"))
    
    # Pattern 2: memory_dump_N.csv  
    pattern2_files = list(dump_dir.glob("memory_dump_[0-9]*.csv"))
    
    # Pattern 3: event_N.csv
    pattern3_files = list(dump_dir.glob("event_*.csv"))
    
    # Pattern 4: dump_N.csv
    pattern4_files = list(dump_dir.glob("dump_*.csv"))
    
    if pattern1_files:
        dump_files = pattern1_files
        print(f"📂 Found {len(dump_files)} files matching 'memory_dump_event_*.csv'")
    elif pattern2_files:
        dump_files = pattern2_files  
        print(f"📂 Found {len(dump_files)} files matching 'memory_dump_*.csv'")
    elif pattern3_files:
        dump_files = pattern3_files
        print(f"📂 Found {len(dump_files)} files matching 'event_*.csv'")
    elif pattern4_files:
        dump_files = pattern4_files
        print(f"📂 Found {len(dump_files)} files matching 'dump_*.csv'")
    else:
        # Fallback: any CSV file
        all_csv = list(dump_dir.glob("*.csv"))
        print(f"📂 No standard pattern found, found {len(all_csv)} CSV files:")
        for f in all_csv[:10]:  # Show first 10
            print(f"  {f.name}")
        
        if not all_csv:
            print("❌ No CSV files found in directory")
            return None
        
        dump_files = all_csv
    
    # Sort files by event number extracted from filename
    def extract_event_num(filename):
        """Extract event number from filename"""
        import re
        # Try to find number in filename
        numbers = re.findall(r'\d+', filename.stem)
        if numbers:
            return int(numbers[-1])  # Use last number found
        return 0
    
    dump_files.sort(key=extract_event_num)
    
    # Load each memory dump file
    memory_states = {}
    
    for file_idx, dump_file in enumerate(dump_files):
        try:
            print(f"  📄 Loading {dump_file.name}...")
            df = pd.read_csv(dump_file)
            
            # Create memory array for this event
            memory = np.zeros((IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.int8)
            
            # Check if this file has the expected columns
            expected_cols = ['x', 'y'] + [f'ch{i}' for i in range(CHANNELS)]
            missing_cols = [col for col in expected_cols if col not in df.columns]
            
            if missing_cols:
                print(f"    ⚠️  Missing columns: {missing_cols}")
                continue
            
            # Load data from this file
            for _, row in df.iterrows():
                try:
                    x, y = int(row['x']), int(row['y'])
                    if 0 <= x < IMG_WIDTH and 0 <= y < IMG_HEIGHT:
                        for ch in range(CHANNELS):
                            memory[y, x, ch] = int(row[f'ch{ch}'])
                except (ValueError, KeyError) as e:
                    # Skip header rows or invalid data silently
                    continue
            
            # Store with correct event index mapping
            # SystemVerilog dumps: memory_dump_event_1.csv = after event 0
            # So file index should be event_number - 1
            event_num = extract_event_num(dump_file)
            if event_num > 0:  # Only process files with event numbers >= 1
                memory_states[event_num - 1] = memory  # Map event_N.csv to PyTorch event N-1
            else:
                memory_states[file_idx] = memory  # Fallback to file order
            
            # Debug info
            nonzero_count = np.count_nonzero(memory)
            mapped_event = event_num - 1 if event_num > 0 else file_idx
            print(f"    ✅ File {dump_file.name} → PyTorch Event {mapped_event}: {len(df)} entries, {nonzero_count} non-zero memory values")
            
        except Exception as e:
            print(f"    ❌ Error loading {dump_file.name}: {e}")
            continue
    
    print(f"✅ Successfully loaded {len(memory_states)} memory dump files")
    
    if len(memory_states) == 0:
        print("❌ No valid memory dump files could be loaded")
        return None
    
    return memory_states

def compare_sequential_results(pytorch_states, systemverilog_states, events):
    """Compare PyTorch and SystemVerilog memory states event by event"""
    print(f"\n🔍 Comparing {len(pytorch_states)} PyTorch states with SystemVerilog...")
    
    total_errors = 0
    results = []
    
    for event_idx in range(len(pytorch_states)):
        if event_idx not in systemverilog_states:
            print(f"❌ Event {event_idx}: No SystemVerilog data")
            continue
        
        # Convert PyTorch tensor to numpy array
        pytorch_memory = pytorch_states[event_idx][0].permute(1, 2, 0).numpy().astype(np.int8)
        sv_memory = systemverilog_states[event_idx]
        
        # Compare
        differences = (pytorch_memory != sv_memory)
        error_count = np.sum(differences)
        
        # Count non-zero values for accuracy
        mask = (pytorch_memory != 0) | (sv_memory != 0)
        total_compared = np.sum(mask)
        
        if total_compared > 0:
            accuracy = ((total_compared - error_count) / total_compared) * 100
        else:
            accuracy = 100.0
        
        total_errors += error_count
        
        # Show results for this event
        if error_count == 0:
            status = "✅"
        else:
            status = "❌"
        
        event_x, event_y = events[event_idx] if event_idx < len(events) else ("?", "?")
        print(f"  {status} Event {event_idx} ({event_x},{event_y}): "
              f"{error_count}/{total_compared} errors ({accuracy:.1f}% accurate)")
        
        # Show first few errors for this event
        if error_count > 0 and error_count <= 5:
            diff_coords = np.where(differences)
            for i in range(min(3, len(diff_coords[0]))):
                y, x, ch = diff_coords[0][i], diff_coords[1][i], diff_coords[2][i]
                pt_val = pytorch_memory[y, x, ch]
                sv_val = sv_memory[y, x, ch]
                print(f"    └─ ({x},{y}) Ch{ch}: PyTorch={pt_val}, SV={sv_val}")
        
        results.append({
            'event_idx': event_idx,
            'errors': error_count,
            'total': total_compared,
            'accuracy': accuracy
        })
    
    # Overall summary
    events_compared = len(results)
    perfect_events = sum(1 for r in results if r['errors'] == 0)
    
    print(f"\n📊 SEQUENTIAL VERIFICATION RESULTS:")
    print(f"   Events compared: {events_compared}")
    print(f"   Perfect matches: {perfect_events}/{events_compared}")
    print(f"   Total errors across all events: {total_errors}")
    
    if total_errors == 0:
        print(f"\n🎉 PERFECT SEQUENTIAL MATCH!")
        print(f"   Your SystemVerilog processes events exactly like PyTorch conv2d!")
        return True
    else:
        print(f"\n❌ SEQUENTIAL VERIFICATION FAILED")
        print(f"   {total_errors} total differences found across all events")
        return False

def save_sequential_debug(pytorch_states, systemverilog_states, events, filename="sequential_debug.txt"):
    """Save detailed sequential debug information"""
    with open(filename, 'w') as f:
        f.write("Sequential Event-by-Event Verification Debug\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Events: {events}\n\n")
        
        for event_idx in range(len(pytorch_states)):
            f.write(f"Event {event_idx}:\n")
            if event_idx < len(events):
                x, y = events[event_idx]
                f.write(f"  Spike at: ({x}, {y})\n")
            
            if event_idx in systemverilog_states:
                pytorch_memory = pytorch_states[event_idx][0].permute(1, 2, 0).numpy().astype(np.int8)
                sv_memory = systemverilog_states[event_idx]
                
                # Show non-zero values
                f.write("  PyTorch non-zero values:\n")
                pt_nonzero = np.where(pytorch_memory != 0)
                for i in range(len(pt_nonzero[0])):
                    y, x, ch = pt_nonzero[0][i], pt_nonzero[1][i], pt_nonzero[2][i]
                    val = pytorch_memory[y, x, ch]
                    f.write(f"    ({x},{y}) Ch{ch}: {val}\n")
                
                f.write("  SystemVerilog non-zero values:\n")
                sv_nonzero = np.where(sv_memory != 0)
                for i in range(len(sv_nonzero[0])):
                    y, x, ch = sv_nonzero[0][i], sv_nonzero[1][i], sv_nonzero[2][i]
                    val = sv_memory[y, x, ch]
                    f.write(f"    ({x},{y}) Ch{ch}: {val}\n")
                
                # Show differences
                differences = (pytorch_memory != sv_memory)
                if np.any(differences):
                    f.write("  Differences:\n")
                    diff_coords = np.where(differences)
                    for i in range(len(diff_coords[0])):
                        y, x, ch = diff_coords[0][i], diff_coords[1][i], diff_coords[2][i]
                        pt_val = pytorch_memory[y, x, ch]
                        sv_val = sv_memory[y, x, ch]
                        f.write(f"    ({x},{y}) Ch{ch}: PyTorch={pt_val}, SV={sv_val}\n")
            else:
                f.write("  No SystemVerilog data for this event\n")
            
            f.write("\n")
    
    print(f"📄 Sequential debug saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description='Sequential event-by-event conv2d verification')
    parser.add_argument('dump_directory', help='Directory containing SystemVerilog memory dump files')
    parser.add_argument('-e', '--events', required=True, help='Events file')
    parser.add_argument('--debug', help='Save debug file')
    parser.add_argument('--show-weights', action='store_true', help='Show kernel weights')
    parser.add_argument('--list-files', action='store_true', help='List found dump files and exit')
    
    args = parser.parse_args()
    
    print("🔄 Sequential Event-by-Event Conv2d Verification")
    print("=" * 50)
    
    # List files if requested
    if args.list_files:
        dump_dir = Path(args.dump_directory)
        if dump_dir.exists():
            print(f"📂 Files in {dump_dir}:")
            csv_files = list(dump_dir.glob("*.csv"))
            for f in sorted(csv_files):
                size_kb = f.stat().st_size / 1024
                print(f"  {f.name} ({size_kb:.1f} KB)")
        else:
            print(f"❌ Directory not found: {dump_dir}")
        return 0
    
    # Load SystemVerilog weights
    sv_weights = load_systemverilog_weights()
    if args.show_weights:
        print("📊 SystemVerilog kernel weights:")
        for pos in range(9):
            row, col = pos // 3, pos % 3
            print(f"  Position ({row},{col}): {sv_weights[pos]}")
        print()
    
    # Create conv2d layer
    conv_layer = create_conv2d_layer(sv_weights)
    print(f"✅ Created conv2d layer: {conv_layer}")
    
    # Load events
    events = load_events(args.events)
    print(f"📝 Loaded {len(events)} events: {events}")
    
    # Generate PyTorch memory states sequentially
    pytorch_states = generate_sequential_memory_states(events, conv_layer)
    
    # Load SystemVerilog memory states from directory
    sv_states = load_systemverilog_memory_states(args.dump_directory)
    if sv_states is None:
        return 1
    
    # Compare sequential results
    success = compare_sequential_results(pytorch_states, sv_states, events)
    
    # Save debug info if requested
    if args.debug:
        save_sequential_debug(pytorch_states, sv_states, events, args.debug)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())