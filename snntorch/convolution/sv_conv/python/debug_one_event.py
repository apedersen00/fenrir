#!/usr/bin/env python3
"""
Simple test for Docker PyTorch environment
Verifies that nn.Conv2d works with our spike data
"""

import torch
import torch.nn as nn
import numpy as np

def test_pytorch_environment():
    """Test basic PyTorch functionality"""
    print("🧠 Testing PyTorch Environment")
    print("=" * 30)
    
    # Check PyTorch version and CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Test basic tensor operations
    x = torch.randn(2, 3)
    print(f"✅ Basic tensor operations working: {x.shape}")
    
    return True

def test_conv2d_with_spikes():
    """Test nn.Conv2d with spike-like data"""
    print("\n🎯 Testing nn.Conv2d with Spike Data")
    print("=" * 35)
    
    # Create test spike data (32x32 image)
    spike_image = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
    
    # Add some test spikes
    test_events = [(5, 5), (10, 10), (15, 15)]
    for x, y in test_events:
        spike_image[0, 0, y, x] = 1.0
    
    print(f"Created spike image with {len(test_events)} spikes")
    
    # Create simple conv2d layer
    conv = nn.Conv2d(
        in_channels=1,
        out_channels=6, 
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    
    # Set some test weights (3x3 kernel)
    with torch.no_grad():
        # Simple edge detection weights for testing
        test_weights = torch.tensor([
            [[-1, -1, -1],
             [ 0,  0,  0], 
             [ 1,  1,  1]]
        ], dtype=torch.float32)
        
        # Replicate for all 6 output channels
        for ch in range(6):
            conv.weight.data[ch, 0] = test_weights * (ch + 1)  # Different weights per channel
    
    print(f"✅ Created conv2d layer: {conv}")
    
    # Apply convolution
    with torch.no_grad():
        output = conv(spike_image)
        
        # Apply saturation like SystemVerilog (-32 to +31)
        output = torch.clamp(output, min=-32, max=31)
    
    print(f"✅ Convolution output shape: {output.shape}")
    print(f"✅ Non-zero outputs: {torch.count_nonzero(output)}")
    print(f"✅ Output range: [{output.min():.1f}, {output.max():.1f}]")
    
    # Show some output values
    print("\n📊 Sample output values around spike locations:")
    for x, y in test_events:
        for ch in range(3):  # Show first 3 channels
            val = output[0, ch, y, x].item()
            print(f"  Spike ({x},{y}) Ch{ch}: {val:.1f}")
    
    return True

def test_systemverilog_weights():
    """Test with actual SystemVerilog weights"""
    print("\n⚙️  Testing with SystemVerilog Weights")
    print("=" * 35)
    
    # SystemVerilog kernel weights
    sv_weights = np.array([
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
    
    # Create conv layer with SystemVerilog weights
    conv = nn.Conv2d(1, 6, 3, stride=1, padding=1, bias=False)
    
    with torch.no_grad():
        weight_tensor = torch.zeros(6, 1, 3, 3)
        for ch in range(6):
            for pos in range(9):
                row, col = pos // 3, pos % 3
                weight_tensor[ch, 0, row, col] = float(sv_weights[pos, ch])  # Convert numpy to Python float
        conv.weight.data = weight_tensor
    
    print(f"✅ Loaded SystemVerilog weights into conv2d")
    
    # Test with a single spike
    spike_image = torch.zeros(1, 1, 32, 32)
    spike_image[0, 0, 16, 16] = 1.0  # Center spike
    
    with torch.no_grad():
        output = conv(spike_image)
        output = torch.clamp(output, min=-32, max=31)
    
    print(f"✅ Applied SystemVerilog convolution")
    print(f"📊 Non-zero outputs: {torch.count_nonzero(output)}")
    
    # Show 3x3 region around center spike
    print("\n📍 3x3 region around center spike (16,16):")
    for ch in range(6):
        print(f"Channel {ch}:")
        region = output[0, ch, 15:18, 15:18]
        for row in range(3):
            values = [f"{region[row, col].item():3.0f}" for col in range(3)]
            print(f"  {' '.join(values)}")
    
    return True

def create_test_events_file():
    """Create a simple test events file"""
    events = [(5, 5), (10, 10), (15, 15), (20, 20)]
    
    with open("test_events.txt", "w") as f:
        f.write("// Test events for Docker verification\n")
        for x, y in events:
            f.write(f"{x},{y}\n")
    
    print(f"✅ Created test_events.txt with {len(events)} events")
    return True

def main():
    print("🐳 Simple Docker PyTorch Test for SNN Verification")
    print("=" * 55)
    
    tests = [
        ("PyTorch Environment", test_pytorch_environment),
        ("Conv2d with Spikes", test_conv2d_with_spikes), 
        ("SystemVerilog Weights", test_systemverilog_weights),
        ("Test Events File", create_test_events_file)
    ]
    
    passed = 0
    for name, test_func in tests:
        try:
            print(f"\n🔧 Running: {name}")
            if test_func():
                print(f"✅ {name}: PASSED")
                passed += 1
            else:
                print(f"❌ {name}: FAILED")
        except Exception as e:
            print(f"❌ {name}: ERROR - {e}")
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Ready for verification.")
        print("\nNext steps:")
        print("1. Copy simulation files: python docker_file_copy.py copy")
        print("2. Run verification: python simple_conv_verification.py mem_dumps/memory_dumps.csv -e test_events.txt")
    else:
        print(f"\n❌ Some tests failed. Check your Docker PyTorch setup.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())