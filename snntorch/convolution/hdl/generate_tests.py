#!/usr/bin/env python3
"""
Simple test vector generator for SNN verification
Usage: python3 generate_tests.py
"""

import numpy as np
import os

def generate_simple_test():
    """Generate a simple test case with realistic event counts"""
    
    # Configuration (matching VHDL)
    IMG_WIDTH = 16
    IMG_HEIGHT = 16
    CHANNELS_OUT = 4
    NEURON_BIT_WIDTH = 9
    NUM_EVENTS = 120  # Realistic number of events for one timestep
    
    # Create test vectors directory
    os.makedirs("test_vectors", exist_ok=True)
    
    print(f"Generating realistic test vectors for {NUM_EVENTS} events...")
    print("Processing flow: events → convolution → timestep_flag → pooling → spikes")
    
    # Generate realistic event pattern: clustered activity + some noise
    events = []
    
    # Clustered activity (80% of events)
    cluster_centers = [(4, 4), (12, 4), (4, 12), (12, 12), (8, 8)]
    num_clustered = int(NUM_EVENTS * 0.8)
    
    for i in range(num_clustered):
        # Pick a random cluster center
        center_x, center_y = cluster_centers[i % len(cluster_centers)]
        
        # Add noise around center (2-pixel radius)
        noise_x = np.random.randint(-2, 3)
        noise_y = np.random.randint(-2, 3)
        
        x = max(0, min(IMG_WIDTH - 1, center_x + noise_x))
        y = max(0, min(IMG_HEIGHT - 1, center_y + noise_y))
        
        events.append((x, y))
    
    # Background noise (20% of events)
    num_noise = NUM_EVENTS - num_clustered
    for i in range(num_noise):
        x = np.random.randint(0, IMG_WIDTH)
        y = np.random.randint(0, IMG_HEIGHT)
        events.append((x, y))
    
    print(f"Generated {len(events)} events:")
    print(f"  - {num_clustered} clustered events around {len(cluster_centers)} centers")
    print(f"  - {num_noise} background noise events")
    
    # Save input events
    with open("test_vectors/input_events.txt", "w") as f:
        f.write("# Input events for single timestep processing\n")
        f.write(f"# Total events: {len(events)}\n")
        f.write("# Format: XXYY (hex coordinates)\n")
        f.write("# Processing: All events → convolution → timestep_flag → pooling → spike_output\n")
        for i, (x, y) in enumerate(events):
            f.write(f"{x:02X}{y:02X}  # Event {i}: ({x}, {y})\n")
    
    # Create expected outputs (simplified - for full verification use python_reference_model.py)
    with open("test_vectors/expected_spikes.txt", "w") as f:
        f.write("# Expected spike events after pooling: XXYY SS\n")
        f.write("# These are simplified expected results\n")
        f.write("# For exact verification, use: python3 python_reference_model.py\n")
        # Based on clustered activity, expect spikes in those regions
        f.write("0202 0F  # Window (2,2) - high activity expected\n")
        f.write("0606 05  # Window (6,6) - moderate activity\n")
        f.write("0404 0A  # Window (4,4) - mixed activity\n")
    
    # Create kernel weights information
    with open("test_vectors/kernel_weights.txt", "w") as f:
        f.write("# Kernel weights configuration\n")
        f.write("# These match the VHDL implementation pattern\n")
        f.write("# Weight(pos, ch) = -(pos+ch+1) if (pos+ch)%3==0 else (pos+ch+1)\n")
        f.write("# Kernel size: 3x3, Channels: 4\n")
        f.write("#\n")
        f.write("# Position mapping (3x3 kernel):\n")
        f.write("# 0 1 2\n") 
        f.write("# 3 4 5\n")
        f.write("# 6 7 8\n")
        for pos in range(9):
            ky, kx = pos // 3, pos % 3
            f.write(f"# Pos {pos} ({ky},{kx}): ")
            for ch in range(CHANNELS_OUT):
                if (pos + ch) % 3 == 0:
                    weight = -(pos + ch + 1)
                else:
                    weight = pos + ch + 1
                f.write(f"ch{ch}={weight:3d} ")
            f.write("\n")
    
    # Create configuration file
    with open("test_vectors/config.txt", "w") as f:
        f.write("# Simple test configuration\n")
        f.write(f"IMG_WIDTH={IMG_WIDTH}\n")
        f.write(f"IMG_HEIGHT={IMG_HEIGHT}\n")
        f.write(f"CHANNELS_OUT={CHANNELS_OUT}\n")
        f.write(f"NUM_EVENTS={len(events)}\n")
        f.write(f"POOL_SIZE=2\n")
        f.write(f"KERNEL_SIZE=3\n")
        f.write("# Processing flow:\n")
        f.write("#   1. Load all events into FIFO\n")
        f.write("#   2. Assert timestep_flag\n") 
        f.write("#   3. Events processed through convolution\n")
        f.write("#   4. Pooling automatically starts after convolution\n")
        f.write("#   5. Spike events generated for active windows\n")
    
    print("✓ Test vectors generated in test_vectors/")
    print("  - input_events.txt: Input event coordinates")
    print("  - expected_spikes.txt: Expected spike outputs (simplified)")
    print("  - kernel_weights.txt: Kernel weight configuration")
    print("  - config.txt: Test configuration")
    
    # Print summary
    print(f"\nTest summary:")
    print(f"  Input events: {len(events)} (realistic for single timestep)")
    print(f"  Image size: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"  Channels: {CHANNELS_OUT}")
    print(f"  Processing: All events → convolution → pooling → spikes")
    print(f"")
    print(f"For complete bit-accurate verification:")
    print(f"  python3 python_reference_model.py")

if __name__ == "__main__":
    np.random.seed(42)  # Reproducible results
    generate_simple_test()