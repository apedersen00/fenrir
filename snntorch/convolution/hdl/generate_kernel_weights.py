#!/usr/bin/env python3
"""
generate_kernel_weights.py

A script to generate VHDL packages with configurable kernel weights for SNN convolution modules.
Supports various weight initialization strategies for different SNN applications.
"""

import numpy as np
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

class WeightType(Enum):
    RANDOM = "random"
    GABOR = "gabor"
    EDGE_DETECTION = "edge_detection"
    GAUSSIAN = "gaussian"
    SOBEL = "sobel"
    IDENTITY = "identity"
    CUSTOM = "custom"
    SPARSE_RANDOM = "sparse_random"

class KernelWeightGenerator:
    """Generate kernel weights for SNN convolution layers"""
    
    def __init__(self, kernel_size: int = 3, channels_out: int = 4, 
                 bits_per_weight: int = 9, seed: int = 42):
        self.kernel_size = kernel_size
        self.channels_out = channels_out
        self.bits_per_weight = bits_per_weight
        self.max_weight = (1 << (bits_per_weight - 1)) - 1  # Max positive signed value
        self.min_weight = -(1 << (bits_per_weight - 1))     # Min negative signed value
        np.random.seed(seed)
        
    def saturate_weight(self, weight: float) -> int:
        """Saturate weight to fit in specified bit width"""
        return int(np.clip(weight, self.min_weight, self.max_weight))
    
    def generate_random_weights(self, sparsity: float = 0.0, 
                              weight_range: Tuple[int, int] = None) -> np.ndarray:
        """Generate random weights with optional sparsity"""
        if weight_range is None:
            weight_range = (self.min_weight // 4, self.max_weight // 4)
        
        weights = np.random.randint(weight_range[0], weight_range[1] + 1, 
                                   (self.kernel_size, self.kernel_size, self.channels_out))
        
        # Apply sparsity (set some weights to zero)
        if sparsity > 0:
            mask = np.random.random(weights.shape) < sparsity
            weights[mask] = 0
            
        return weights.astype(np.int32)
    
    def generate_gabor_weights(self, orientations: List[float] = None,
                              frequencies: List[float] = None) -> np.ndarray:
        """Generate Gabor filter weights for edge/texture detection"""
        if orientations is None:
            orientations = [0, 45, 90, 135]  # degrees
        if frequencies is None:
            frequencies = [0.3] * len(orientations)
            
        # Ensure we have enough orientations for all channels
        while len(orientations) < self.channels_out:
            orientations.extend(orientations)
        while len(frequencies) < self.channels_out:
            frequencies.extend(frequencies)
            
        weights = np.zeros((self.kernel_size, self.kernel_size, self.channels_out), dtype=np.int32)
        center = self.kernel_size // 2
        
        for ch in range(self.channels_out):
            theta = np.radians(orientations[ch])
            freq = frequencies[ch]
            
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    x = j - center
                    y = i - center
                    
                    # Rotate coordinates
                    x_rot = x * np.cos(theta) + y * np.sin(theta)
                    y_rot = -x * np.sin(theta) + y * np.cos(theta)
                    
                    # Gabor function
                    gaussian = np.exp(-(x_rot**2 + y_rot**2) / (2 * 0.5**2))
                    sinusoid = np.cos(2 * np.pi * freq * x_rot)
                    gabor_val = gaussian * sinusoid
                    
                    # Scale and saturate
                    scaled_val = gabor_val * (self.max_weight // 4)
                    weights[i, j, ch] = self.saturate_weight(scaled_val)
                    
        return weights
    
    def generate_edge_detection_weights(self) -> np.ndarray:
        """Generate edge detection kernels (Sobel, Prewitt, etc.)"""
        weights = np.zeros((self.kernel_size, self.kernel_size, self.channels_out), dtype=np.int32)
        
        if self.kernel_size == 3:
            # Standard 3x3 edge detection kernels
            kernels = {
                0: np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),      # Horizontal edge
                1: np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),      # Vertical edge  
                2: np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]),      # Diagonal edge
                3: np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])       # Other diagonal
            }
        else:
            # For other sizes, generate simple patterns
            kernels = {}
            for ch in range(self.channels_out):
                kernel = np.random.choice([-1, 0, 1], (self.kernel_size, self.kernel_size))
                kernels[ch] = kernel
        
        for ch in range(self.channels_out):
            kernel_idx = ch % len(kernels)
            weights[:, :, ch] = kernels[kernel_idx] * (self.max_weight // 8)
            
        return weights
    
    def generate_gaussian_weights(self, sigma: float = 1.0) -> np.ndarray:
        """Generate Gaussian blur kernels"""
        weights = np.zeros((self.kernel_size, self.kernel_size, self.channels_out), dtype=np.int32)
        center = self.kernel_size // 2
        
        # Generate base Gaussian kernel
        gaussian_kernel = np.zeros((self.kernel_size, self.kernel_size))
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                x = j - center
                y = i - center
                gaussian_kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        # Normalize and scale
        gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel * (self.max_weight // 2)
        
        # Apply to all channels with slight variations
        for ch in range(self.channels_out):
            variation = 1.0 + (ch * 0.1)  # Slight variation per channel
            weights[:, :, ch] = np.round(gaussian_kernel * variation).astype(np.int32)
            
        return weights
    
    def generate_identity_weights(self) -> np.ndarray:
        """Generate identity/passthrough kernels"""
        weights = np.zeros((self.kernel_size, self.kernel_size, self.channels_out), dtype=np.int32)
        center = self.kernel_size // 2
        
        for ch in range(self.channels_out):
            weights[center, center, ch] = self.max_weight // 4
            
        return weights
    
    def load_custom_weights(self, filepath: str) -> np.ndarray:
        """Load weights from file (numpy .npy, .npz, or JSON)"""
        path = Path(filepath)
        
        if path.suffix == '.npy':
            weights = np.load(filepath)
        elif path.suffix == '.npz':
            archive = np.load(filepath)
            weights = archive['weights']  # Assumes key 'weights'
        elif path.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            weights = np.array(data['weights'])
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Verify dimensions
        expected_shape = (self.kernel_size, self.kernel_size, self.channels_out)
        if weights.shape != expected_shape:
            raise ValueError(f"Weight shape {weights.shape} doesn't match expected {expected_shape}")
        
        # Saturate to bit width
        weights = np.clip(weights, self.min_weight, self.max_weight)
        return weights.astype(np.int32)
    
    def generate_weights(self, weight_type: WeightType, **kwargs) -> np.ndarray:
        """Generate weights based on specified type"""
        if weight_type == WeightType.RANDOM:
            return self.generate_random_weights(**kwargs)
        elif weight_type == WeightType.GABOR:
            return self.generate_gabor_weights(**kwargs)
        elif weight_type == WeightType.EDGE_DETECTION:
            return self.generate_edge_detection_weights()
        elif weight_type == WeightType.GAUSSIAN:
            return self.generate_gaussian_weights(**kwargs)
        elif weight_type == WeightType.SOBEL:
            return self.generate_edge_detection_weights()  # Same as edge detection
        elif weight_type == WeightType.IDENTITY:
            return self.generate_identity_weights()
        elif weight_type == WeightType.SPARSE_RANDOM:
            return self.generate_random_weights(sparsity=kwargs.get('sparsity', 0.3))
        elif weight_type == WeightType.CUSTOM:
            return self.load_custom_weights(kwargs['filepath'])
        else:
            raise ValueError(f"Unknown weight type: {weight_type}")

def generate_vhdl_package(weights: np.ndarray, package_name: str = "kernel_weights_pkg",
                         kernel_size: int = 3, channels_out: int = 4, 
                         bits_per_weight: int = 9) -> str:
    """Generate VHDL package with kernel weights"""
    
    vhdl_code = f"""----------------------------------------------------------------------------------------------------
--  Auto-generated VHDL Package: {package_name}
--  Generated by generate_kernel_weights.py
--  
--  Contains kernel weights for SNN convolution module
--  Kernel size: {kernel_size}x{kernel_size}, Channels: {channels_out}, Weight bits: {bits_per_weight}
----------------------------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package {package_name} is

    -- Kernel configuration constants
    constant KERNEL_SIZE_CFG     : integer := {kernel_size};
    constant CHANNELS_OUT_CFG    : integer := {channels_out};
    constant BITS_PER_WEIGHT_CFG : integer := {bits_per_weight};
    
    -- Kernel weight type definition
    type kernel_weights_t is array (0 to KERNEL_SIZE_CFG**2 - 1, 0 to CHANNELS_OUT_CFG - 1) 
         of signed(BITS_PER_WEIGHT_CFG - 1 downto 0);
    
    -- Kernel weight constants
    constant KERNEL_WEIGHTS : kernel_weights_t := (
"""
    
    # Generate weight initialization
    for pos in range(kernel_size * kernel_size):
        row = pos // kernel_size
        col = pos % kernel_size
        vhdl_code += f"        {pos:2d} => ("  # Position comment
        
        for ch in range(channels_out):
            weight_val = int(weights[row, col, ch])
            if ch == channels_out - 1:
                vhdl_code += f"{weight_val:4d}"
            else:
                vhdl_code += f"{weight_val:4d}, "
        
        if pos == kernel_size * kernel_size - 1:
            vhdl_code += f"), -- Pos({row},{col})\n"
        else:
            vhdl_code += f"), -- Pos({row},{col})\n"
    
    vhdl_code += """    );
    
    -- Helper function to get weight value
    function get_kernel_weight(
        kernel_pos : integer;
        channel : integer
    ) return signed;
    
end package """ + package_name + """;

package body """ + package_name + """ is

    function get_kernel_weight(
        kernel_pos : integer;
        channel : integer
    ) return signed is
    begin
        if kernel_pos >= 0 and kernel_pos < KERNEL_SIZE_CFG**2 and
           channel >= 0 and channel < CHANNELS_OUT_CFG then
            return KERNEL_WEIGHTS(kernel_pos, channel);
        else
            return to_signed(0, BITS_PER_WEIGHT_CFG);
        end if;
    end function;

end package body """ + package_name + """;
"""
    
    return vhdl_code

def save_weights_metadata(weights: np.ndarray, filepath: str, weight_type: str, **kwargs):
    """Save weight metadata for analysis and verification"""
    metadata = {
        'weight_type': weight_type,
        'shape': weights.shape,
        'min_weight': int(np.min(weights)),
        'max_weight': int(np.max(weights)),
        'mean_weight': float(np.mean(weights)),
        'std_weight': float(np.std(weights)),
        'num_zeros': int(np.sum(weights == 0)),
        'sparsity': float(np.sum(weights == 0) / weights.size),
        'generation_params': kwargs
    }
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)

def visualize_weights(weights: np.ndarray, output_dir: str):
    """Generate simple text visualization of weights"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for ch in range(weights.shape[2]):
        with open(output_path / f"channel_{ch}_weights.txt", 'w') as f:
            f.write(f"Channel {ch} Kernel Weights:\n")
            f.write("=" * 30 + "\n")
            for row in range(weights.shape[0]):
                for col in range(weights.shape[1]):
                    f.write(f"{weights[row, col, ch]:4d} ")
                f.write("\n")
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(
        description="Generate VHDL packages with configurable kernel weights for SNN convolution"
    )
    parser.add_argument("--type", type=str, choices=[t.value for t in WeightType], 
                       default="random", help="Weight generation type")
    parser.add_argument("--kernel-size", type=int, default=3, help="Kernel size (NxN)")
    parser.add_argument("--channels", type=int, default=4, help="Number of output channels")
    parser.add_argument("--bits", type=int, default=9, help="Bits per weight")
    parser.add_argument("--package-name", type=str, default="kernel_weights_pkg", 
                       help="VHDL package name")
    parser.add_argument("--output-dir", type=str, default="generated_weights", 
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Weight-specific parameters
    parser.add_argument("--sparsity", type=float, default=0.0, 
                       help="Sparsity for sparse random weights (0.0-1.0)")
    parser.add_argument("--custom-file", type=str, help="Path to custom weight file")
    parser.add_argument("--sigma", type=float, default=1.0, help="Sigma for Gaussian weights")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize generator
    generator = KernelWeightGenerator(
        kernel_size=args.kernel_size,
        channels_out=args.channels,
        bits_per_weight=args.bits,
        seed=args.seed
    )
    
    # Generate weights
    weight_type = WeightType(args.type)
    kwargs = {}
    
    if weight_type == WeightType.SPARSE_RANDOM:
        kwargs['sparsity'] = args.sparsity
    elif weight_type == WeightType.CUSTOM:
        if not args.custom_file:
            raise ValueError("Custom weight file path required for custom type")
        kwargs['filepath'] = args.custom_file
    elif weight_type == WeightType.GAUSSIAN:
        kwargs['sigma'] = args.sigma
    
    print(f"Generating {weight_type.value} weights...")
    weights = generator.generate_weights(weight_type, **kwargs)
    
    # Generate VHDL package
    print("Generating VHDL package...")
    vhdl_code = generate_vhdl_package(
        weights, args.package_name, args.kernel_size, args.channels, args.bits
    )
    
    # Save files
    vhdl_file = output_dir / f"{args.package_name}.vhd"
    with open(vhdl_file, 'w') as f:
        f.write(vhdl_code)
    
    # Save numpy weights for reference
    np.save(output_dir / "weights.npy", weights)
    
    # Save metadata
    save_weights_metadata(weights, output_dir / "metadata.json", args.type, **kwargs)
    
    # Generate visualizations
    visualize_weights(weights, str(output_dir))
    
    print(f"\nGenerated files in {output_dir}:")
    print(f"  {args.package_name}.vhd - VHDL package")
    print(f"  weights.npy - NumPy weights for reference")
    print(f"  metadata.json - Weight statistics and parameters")
    print(f"  channel_*_weights.txt - Text visualization")
    
    print(f"\nWeight Statistics:")
    print(f"  Shape: {weights.shape}")
    print(f"  Range: [{np.min(weights)}, {np.max(weights)}]")
    print(f"  Mean: {np.mean(weights):.2f}")
    print(f"  Std: {np.std(weights):.2f}")
    print(f"  Sparsity: {np.sum(weights == 0) / weights.size:.1%}")

if __name__ == "__main__":
    main()