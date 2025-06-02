#!/usr/bin/env python3
"""
kernel_weight_workflow.py

Example workflow script showing how to generate and use different kernel weight configurations
for SNN convolution testing. Demonstrates integration with your existing testbench setup.
"""

import os
import subprocess
import numpy as np
from pathlib import Path

def run_weight_generation(weight_type, config, output_dir):
    """Run the weight generator with specified configuration"""
    cmd = [
        "python3", "generate_kernel_weights.py",
        "--type", weight_type,
        "--kernel-size", str(config['kernel_size']),
        "--channels", str(config['channels']),
        "--bits", str(config['bits']),
        "--package-name", config['package_name'],
        "--output-dir", output_dir,
        "--seed", str(config.get('seed', 42))
    ]
    
    # Add type-specific parameters
    if weight_type == "sparse_random":
        cmd.extend(["--sparsity", str(config.get('sparsity', 0.3))])
    elif weight_type == "gaussian":
        cmd.extend(["--sigma", str(config.get('sigma', 1.0))])
    elif weight_type == "custom":
        cmd.extend(["--custom-file", config['custom_file']])
    
    print(f"Generating {weight_type} weights...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Weight generation successful")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Weight generation failed")
        print(f"Error: {e.stderr}")
        return False

def copy_weights_to_project(source_dir, target_file="kernel_weights_pkg.vhd"):
    """Copy generated weights package to project directory"""
    source_file = Path(source_dir) / "kernel_weights_pkg.vhd"
    target_path = Path(target_file)
    
    if source_file.exists():
        # Copy the file
        import shutil
        shutil.copy2(source_file, target_path)
        print(f"✅ Copied weights package to {target_path}")
        return True
    else:
        print(f"❌ Source file {source_file} not found")
        return False

def run_testbench(testbench_name="tb_convolution"):
    """Run VUnit testbench with new weights"""
    print(f"Running testbench: {testbench_name}")
    
    # You can adapt this to your specific testbench runner
    cmd = ["python3", "run.py", f"*{testbench_name}*"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Testbench passed")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Testbench failed")
        print(f"Error output: {e.stderr}")
        return False

def create_custom_weights_example():
    """Create a custom weights file for demonstration"""
    # Create a simple custom weight pattern
    kernel_size = 3
    channels = 4
    weights = np.zeros((kernel_size, kernel_size, channels), dtype=np.int32)
    
    # Create a specific pattern - center-surround for each channel
    center = kernel_size // 2
    for ch in range(channels):
        # Different patterns per channel
        if ch == 0:  # Horizontal edge
            weights[:, :, ch] = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        elif ch == 1:  # Vertical edge
            weights[:, :, ch] = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        elif ch == 2:  # Center-surround
            weights[:, :, ch] = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        else:  # Identity
            weights[center, center, ch] = 4
    
    # Save as numpy file
    np.save("custom_weights.npy", weights)
    
    # Also save as JSON for easy inspection
    import json
    weights_dict = {
        "weights": weights.tolist(),
        "description": "Custom weights with different patterns per channel",
        "kernel_size": kernel_size,
        "channels": channels
    }
    
    with open("custom_weights.json", "w") as f:
        json.dump(weights_dict, f, indent=2)
    
    print("✅ Created custom weights example (custom_weights.npy and custom_weights.json)")
    return "custom_weights.npy"

def demonstrate_workflow():
    """Demonstrate the complete workflow with different weight types"""
    
    print("🧠 SNN Kernel Weight Configuration Workflow")
    print("=" * 50)
    
    # Base configuration matching your convolution module
    base_config = {
        'kernel_size': 3,
        'channels': 4,
        'bits': 9,
        'package_name': 'kernel_weights_pkg',
        'seed': 42
    }
    
    # Test different weight configurations
    weight_configs = [
        {
            'type': 'edge_detection',
            'description': 'Edge detection kernels (Sobel-like)',
            'config': base_config.copy()
        },
        {
            'type': 'gabor',
            'description': 'Gabor filters for texture detection',
            'config': base_config.copy()
        },
        {
            'type': 'sparse_random',
            'description': 'Sparse random weights (30% zeros)',
            'config': {**base_config, 'sparsity': 0.3}
        },
        {
            'type': 'gaussian',
            'description': 'Gaussian blur kernels',
            'config': {**base_config, 'sigma': 1.0}
        }
    ]
    
    # Create directories for different configurations
    results_dir = Path("kernel_weight_results")
    results_dir.mkdir(exist_ok=True)
    
    successful_configs = []
    
    for i, weight_config in enumerate(weight_configs, 1):
        print(f"\n{i}. Testing {weight_config['description']}")
        print("-" * 40)
        
        # Create output directory for this configuration
        config_dir = results_dir / f"config_{i}_{weight_config['type']}"
        config_dir.mkdir(exist_ok=True)
        
        # Generate weights
        success = run_weight_generation(
            weight_config['type'], 
            weight_config['config'], 
            str(config_dir)
        )
        
        if success:
            # Copy to project directory
            copy_success = copy_weights_to_project(
                str(config_dir), 
                "kernel_weights_pkg.vhd"
            )
            
            if copy_success:
                # Optionally run testbench (commented out for demo)
                # testbench_success = run_testbench("tb_convolution")
                # if testbench_success:
                #     successful_configs.append(weight_config)
                successful_configs.append(weight_config)
                print(f"✅ Configuration {i} completed successfully")
            else:
                print(f"❌ Configuration {i} failed during file copy")
        else:
            print(f"❌ Configuration {i} failed during weight generation")
    
    # Test custom weights
    print(f"\n{len(weight_configs) + 1}. Testing custom weights")
    print("-" * 40)
    
    custom_file = create_custom_weights_example()
    custom_config = {**base_config, 'custom_file': custom_file}
    custom_dir = results_dir / "config_custom"
    custom_dir.mkdir(exist_ok=True)
    
    custom_success = run_weight_generation("custom", custom_config, str(custom_dir))
    if custom_success:
        copy_weights_to_project(str(custom_dir), "kernel_weights_pkg.vhd")
        successful_configs.append({'type': 'custom', 'description': 'Custom weight patterns'})
    
    # Summary
    print(f"\n🎉 Workflow Summary")
    print("=" * 30)
    print(f"Total configurations tested: {len(weight_configs) + 1}")
    print(f"Successful configurations: {len(successful_configs)}")
    
    print(f"\nSuccessful configurations:")
    for config in successful_configs:
        print(f"  ✅ {config['type']}: {config['description']}")
    
    print(f"\nGenerated files in {results_dir}/:")
    for item in results_dir.iterdir():
        if item.is_dir():
            print(f"  📁 {item.name}/")
            for subitem in item.iterdir():
                print(f"    📄 {subitem.name}")
    
    print(f"\nNext steps:")
    print(f"1. The active kernel_weights_pkg.vhd contains the last generated weights")
    print(f"2. Run your VUnit testbenches: python3 run.py '*convolution*'")
    print(f"3. Use convert_to_vivado.py for Vivado-compatible testbenches")
    print(f"4. Synthesize with your preferred FPGA tools")
    
    print(f"\nTo use a specific weight configuration:")
    print(f"  cp kernel_weight_results/config_1_edge_detection/kernel_weights_pkg.vhd .")
    print(f"  # Then recompile and run tests")

def interactive_weight_generation():
    """Interactive mode for generating specific weight configurations"""
    
    print("🔧 Interactive Kernel Weight Generator")
    print("=" * 40)
    
    # Get configuration from user
    try:
        kernel_size = int(input("Kernel size (3, 5, 7) [3]: ") or "3")
        channels = int(input("Number of channels [4]: ") or "4")
        bits = int(input("Bits per weight [9]: ") or "9")
        
        print("\nAvailable weight types:")
        weight_types = [
            "random", "gabor", "edge_detection", "gaussian", 
            "sparse_random", "identity", "custom"
        ]
        for i, wt in enumerate(weight_types, 1):
            print(f"  {i}. {wt}")
        
        choice = input(f"Select weight type (1-{len(weight_types)}) [1]: ") or "1"
        weight_type = weight_types[int(choice) - 1]
        
        package_name = input("Package name [kernel_weights_pkg]: ") or "kernel_weights_pkg"
        output_dir = input("Output directory [generated_weights]: ") or "generated_weights"
        
        # Build configuration
        config = {
            'kernel_size': kernel_size,
            'channels': channels,
            'bits': bits,
            'package_name': package_name,
            'seed': 42
        }
        
        # Get type-specific parameters
        if weight_type == "sparse_random":
            sparsity = float(input("Sparsity (0.0-1.0) [0.3]: ") or "0.3")
            config['sparsity'] = sparsity
        elif weight_type == "gaussian":
            sigma = float(input("Sigma [1.0]: ") or "1.0")
            config['sigma'] = sigma
        elif weight_type == "custom":
            custom_file = input("Custom weights file path: ")
            config['custom_file'] = custom_file
        
        # Generate weights
        success = run_weight_generation(weight_type, config, output_dir)
        
        if success:
            copy_to_project = input("Copy to project directory? (y/n) [y]: ") or "y"
            if copy_to_project.lower() == 'y':
                copy_weights_to_project(output_dir, "kernel_weights_pkg.vhd")
        
    except (ValueError, KeyboardInterrupt) as e:
        print(f"\n❌ Error or cancelled: {e}")

def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_weight_generation()
    else:
        demonstrate_workflow()

if __name__ == "__main__":
    main()