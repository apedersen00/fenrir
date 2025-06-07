#!/usr/bin/env python3
"""
Quick comparison of first event to debug remaining differences
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

def load_systemverilog_weights():
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
    conv = nn.Conv2d(1, 6, 3, stride=1, padding=1, bias=False)
    weight_tensor = torch.zeros(6, 1, 3, 3)
    for ch in range(6):
        for pos in range(9):
            sv_row = pos // 3
            sv_col = pos % 3
            pytorch_row = 2 - sv_row
            pytorch_col = 2 - sv_col
            weight_tensor[ch, 0, pytorch_row, pytorch_col] = float(sv_weights[pos, ch])
    conv.weight.data = weight_tensor
    return conv

def main():
    print("🔍 Quick First Event Comparison")
    print("=" * 40)
    
    # First event: (5,5)
    event_x, event_y = 5, 5
    
    # Create PyTorch result
    sv_weights = load_systemverilog_weights()
    conv = create_conv2d_layer(sv_weights)
    
    image = torch.zeros(1, 1, 32, 32)
    image[0, 0, event_y, event_x] = 1.0
    
    with torch.no_grad():
        pytorch_output = conv(image)
        pytorch_output = torch.clamp(pytorch_output, min=-32, max=31)
    
    pytorch_memory = pytorch_output[0].permute(1, 2, 0).numpy().astype(np.int8)
    pytorch_nonzero = np.count_nonzero(pytorch_memory)
    
    print(f"📊 PyTorch Event 0 - spike at ({event_x},{event_y})")
    print(f"   Non-zero values: {pytorch_nonzero}")
    
    # Load SystemVerilog result (should be memory_dump_event_1.csv)
    try:
        df = pd.read_csv("mem_dumps/memory_dump_event_1.csv")
        sv_memory = np.zeros((32, 32, 6), dtype=np.int8)
        
        for _, row in df.iterrows():
            try:
                x, y = int(row['x']), int(row['y'])
                if 0 <= x < 32 and 0 <= y < 32:
                    for ch in range(6):
                        sv_memory[y, x, ch] = int(row[f'ch{ch}'])
            except (ValueError, KeyError):
                continue
        
        sv_nonzero = np.count_nonzero(sv_memory)
        print(f"📊 SystemVerilog Event 1 file (after event 0)")
        print(f"   Non-zero values: {sv_nonzero}")
        
        # Compare
        differences = (pytorch_memory != sv_memory)
        error_count = np.sum(differences)
        mask = (pytorch_memory != 0) | (sv_memory != 0)
        total_compared = np.sum(mask)
        
        if total_compared > 0:
            accuracy = ((total_compared - error_count) / total_compared) * 100
        else:
            accuracy = 100.0
        
        print(f"🔍 Comparison:")
        print(f"   Total compared: {total_compared}")
        print(f"   Errors: {error_count}")
        print(f"   Accuracy: {accuracy:.1f}%")
        
        # Show 5x5 region around spike
        print(f"\n🎯 5x5 region around spike ({event_x},{event_y}):")
        
        start_y = max(0, event_y - 2)
        end_y = min(32, event_y + 3)
        start_x = max(0, event_x - 2)
        end_x = min(32, event_x + 3)
        
        for ch in range(6):
            print(f"\nChannel {ch}:")
            print("PyTorch:")
            for y in range(start_y, end_y):
                print(f"  {y:2d}: ", end="")
                for x in range(start_x, end_x):
                    val = pytorch_memory[y, x, ch]
                    if val == 0:
                        print("   .", end="")
                    else:
                        print(f"{val:4d}", end="")
                print()
            
            print("SystemVerilog:")
            for y in range(start_y, end_y):
                print(f"  {y:2d}: ", end="")
                for x in range(start_x, end_x):
                    val = sv_memory[y, x, ch]
                    if val == 0:
                        print("   .", end="")
                    else:
                        print(f"{val:4d}", end="")
                print()
            
            print("Differences:")
            for y in range(start_y, end_y):
                print(f"  {y:2d}: ", end="")
                for x in range(start_x, end_x):
                    pt_val = pytorch_memory[y, x, ch]
                    sv_val = sv_memory[y, x, ch]
                    if pt_val == sv_val:
                        print("   .", end="")
                    else:
                        diff = pt_val - sv_val
                        print(f"{diff:+4d}", end="")
                print()
        
        # Show first 10 exact differences
        if error_count > 0:
            print(f"\n🔍 First 10 exact differences:")
            diff_coords = np.where(differences)
            for i in range(min(10, len(diff_coords[0]))):
                y, x, ch = diff_coords[0][i], diff_coords[1][i], diff_coords[2][i]
                pt_val = pytorch_memory[y, x, ch]
                sv_val = sv_memory[y, x, ch]
                print(f"   ({x:2d},{y:2d}) Ch{ch}: PyTorch={pt_val:3d}, SV={sv_val:3d}, diff={pt_val-sv_val:+d}")
        
    except Exception as e:
        print(f"❌ Error loading SystemVerilog data: {e}")

if __name__ == "__main__":
    main()