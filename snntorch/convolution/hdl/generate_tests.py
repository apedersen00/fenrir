#!/usr/bin/env python3
"""
Simple test vector generator for SNN verification
Usage: python3 generate_tests.py
"""

import numpy as np
import os

def generate_simple_test():
    """Generate a simple test case with known expected outputs"""
    
    # Configuration (matching VHDL)
    IMG_WIDTH = 16
    IMG_HEIGHT = 16
    CHANNELS_OUT = 4
    NEURON_BIT_WIDTH = 9
    
    # Create test vectors directory
    os.makedirs("test_vectors", exist_ok=True)
    
    # Simple test: 4 events in corners
    events = [
        (2, 2),   # Top-left area
        (13, 2),  # Top-right area  
        (2, 13),  # Bottom-left area
        (13, 13)  # Bottom-right area
    ]
    
    print(f"Generating test vectors for {len(events)} events...")
    
    # Save input events
    with open("test_vectors/input_events.txt", "w") as f:
        f.write("# Input events: hex format XXYY\n")
        for x, y in events:
            f.write(f"{x:02X}{y:02X}\n")
    
    # Create expected outputs (simplified for initial testing)
    # In a real implementation, you'd run the Python reference model
    
    with open("test_vectors/expected_spikes.txt", "w") as f:
        f.write("# Expected spike events: XXYY SS (coordinates and spike vector in hex)\n")
        # Since we don't run the full model, create plausible expected spikes
        # These are dummy values for testing the verification framework
        f.write("0101 05\n")  # Spike at window (1,1) with channels 0&2 active
        f.write("0606 0A\n")  # Spike at window (6,6) with channels 1&3 active
    
    # Create configuration file
    with open("test_vectors/config.txt", "w") as f:
        f.write("# Test configuration\n")
        f.write(f"IMG_WIDTH={IMG_WIDTH}\n")
        f.write(f"IMG_HEIGHT={IMG_HEIGHT}\n")
        f.write(f"CHANNELS_OUT={CHANNELS_OUT}\n")
        f.write(f"NUM_EVENTS={len(events)}\n")
    
    print("✓ Test vectors generated in test_vectors/")
    print("  - input_events.txt: Input event coordinates")
    print("  - expected_spikes.txt: Expected spike outputs")
    print("  - config.txt: Test configuration")
    
    # Print summary
    print(f"\nTest summary:")
    print(f"  Input events: {len(events)}")
    print(f"  Image size: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"  Channels: {CHANNELS_OUT}")
    
    for i, (x, y) in enumerate(events):
        print(f"  Event {i}: ({x}, {y}) -> 0x{x:02X}{y:02X}")

if __name__ == "__main__":
    generate_simple_test()