#!/usr/bin/env python3
"""
Debug addressing and coordinate system differences
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

def test_coordinate_systems():
    """Test different coordinate system interpretations"""
    
    print("🔍 Testing Coordinate System Mapping")
    print("=" * 50)
    
    # Test event: spike at (5, 5)
    event_x, event_y = 5, 5
    IMG_WIDTH, IMG_HEIGHT = 32, 32
    
    print(f"Event: spike at (x={event_x}, y={event_y})")
    print(f"SystemVerilog BRAM address = y * IMG_WIDTH + x = {event_y} * {IMG_WIDTH} + {event_x} = {event_y * IMG_WIDTH + event_x}")
    
    # Method 1: Standard PyTorch indexing [batch, channels, height, width] = [batch, channels, y, x]
    print(f"\n📊 Method 1: Standard PyTorch [batch, channels, y, x]")
    image1 = torch.zeros(1, 1, IMG_HEIGHT, IMG_WIDTH)
    image1[0, 0, event_y, event_x] = 1.0
    print(f"   PyTorch tensor[0, 0, {event_y}, {event_x}] = 1.0")
    
    # Method 2: Swapped coordinates [batch, channels, x, y] - WRONG but let's test
    print(f"\n📊 Method 2: Swapped coordinates [batch, channels, x, y]")
    image2 = torch.zeros(1, 1, IMG_WIDTH, IMG_HEIGHT)  # Note: swapped dimensions
    image2[0, 0, event_x, event_y] = 1.0
    print(f"   PyTorch tensor[0, 0, {event_x}, {event_y}] = 1.0")
    
    # Show where the spike actually landed in both cases
    print(f"\n🎯 Spike locations in tensors:")
    
    spike_coords1 = torch.nonzero(image1[0, 0])
    if len(spike_coords1) > 0:
        y, x = spike_coords1[0]
        print(f"   Method 1: Spike found at tensor[{y}, {x}] which corresponds to (x={x}, y={y})")
    
    spike_coords2 = torch.nonzero(image2[0, 0])
    if len(spike_coords2) > 0:
        x, y = spike_coords2[0]  # Note: dimensions are swapped in image2
        print(f"   Method 2: Spike found at tensor[{x}, {y}] which corresponds to (x={y}, y={x})")
    
    return image1, image2

def test_memory_array_conversion():
    """Test conversion from PyTorch output to memory array"""
    
    print(f"\n🔄 Testing Memory Array Conversion")
    print("=" * 40)
    
    # Create a test PyTorch output tensor [1, 6, 32, 32]
    test_output = torch.zeros(1, 6, 32, 32)
    
    # Put a test pattern: value 10 at (x=5, y=7) in channel 2
    test_x, test_y, test_ch = 5, 7, 2
    test_output[0, test_ch, test_y, test_x] = 10
    
    print(f"Test: Put value 10 at (x={test_x}, y={test_y}) in channel {test_ch}")
    print(f"PyTorch tensor[0, {test_ch}, {test_y}, {test_x}] = 10")
    
    # Method 1: Standard conversion
    memory1 = test_output[0].permute(1, 2, 0).numpy()  # [channels, height, width] -> [height, width, channels]
    print(f"\nMethod 1: permute(1, 2, 0) -> [height, width, channels]")
    print(f"   memory1[{test_y}, {test_x}, {test_ch}] = {memory1[test_y, test_x, test_ch]}")
    
    # Method 2: Different conversion
    memory2 = test_output[0].permute(2, 1, 0).numpy()  # [channels, height, width] -> [width, height, channels]
    print(f"\nMethod 2: permute(2, 1, 0) -> [width, height, channels]")
    print(f"   memory2[{test_x}, {test_y}, {test_ch}] = {memory2[test_x, test_y, test_ch]}")
    
    return memory1, memory2

def compare_with_systemverilog():
    """Compare coordinate handling with SystemVerilog CSV"""
    
    print(f"\n📁 Comparing with SystemVerilog CSV")
    print("=" * 40)
    
    try:
        # Load the first SystemVerilog dump
        df = pd.read_csv("mem_dumps/memory_dump_event_1.csv")
        
        print(f"SystemVerilog CSV has {len(df)} entries")
        print("First 5 non-header entries:")
        
        count = 0
        for _, row in df.iterrows():
            try:
                x, y = int(row['x']), int(row['y'])
                ch0_val = int(row['ch0'])
                
                # Show first few non-zero entries
                if ch0_val != 0:
                    print(f"   Entry: x={x}, y={y}, ch0={ch0_val}")
                    
                    # Calculate BRAM address like SystemVerilog
                    bram_addr = y * 32 + x
                    print(f"      BRAM address = {y} * 32 + {x} = {bram_addr}")
                    
                    count += 1
                    if count >= 5:
                        break
                        
            except (ValueError, KeyError):
                continue
        
        # Check the specific event location (5,5)
        event_5_5 = df[(df['x'] == 5) & (df['y'] == 5)]
        if len(event_5_5) > 0:
            print(f"\nSystemVerilog data at event location (5,5):")
            row = event_5_5.iloc[0]
            for ch in range(6):
                val = int(row[f'ch{ch}'])
                print(f"   Channel {ch}: {val}")
        else:
            print(f"\nNo SystemVerilog data found at (5,5) - checking nearby...")
            nearby = df[((df['x'] >= 4) & (df['x'] <= 6)) & ((df['y'] >= 4) & (df['y'] <= 6))]
            print(f"Found {len(nearby)} entries near (5,5):")
            for _, row in nearby.iterrows():
                x, y = int(row['x']), int(row['y'])
                ch0_val = int(row['ch0'])
                print(f"   ({x}, {y}): ch0={ch0_val}")
                
    except Exception as e:
        print(f"Error loading SystemVerilog data: {e}")

def main():
    print("🔍 Debugging BRAM Addressing and Coordinate Systems")
    print("=" * 60)
    
    # Test coordinate systems
    image1, image2 = test_coordinate_systems()
    
    # Test memory conversion
    memory1, memory2 = test_memory_array_conversion()
    
    # Compare with SystemVerilog
    compare_with_systemverilog()
    
    print(f"\n💡 Key Questions:")
    print(f"1. Does PyTorch use (y,x) or (x,y) indexing?")
    print(f"2. Does SystemVerilog CSV use (x,y) coordinates correctly?") 
    print(f"3. Is the memory array conversion preserving coordinates?")
    print(f"4. Are we comparing the right memory locations?")

if __name__ == "__main__":
    main()