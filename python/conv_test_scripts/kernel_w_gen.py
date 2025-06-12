#!/usr/bin/env python3
"""
Kernel Weight Generator for SNN Convolution
Generates kernel weights and outputs files for both Vivado and Python mockup
"""

import argparse
import numpy as np
import json
from pathlib import Path
import math

class KernelGenerator:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 bits_per_weight: int = 6):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bits_per_weight = bits_per_weight
        self.total_positions = kernel_size * kernel_size
        
        # Calculate value ranges for signed integers
        self.max_value = (1 << (bits_per_weight - 1)) - 1  # e.g., 31 for 6-bit
        self.min_value = -(1 << (bits_per_weight - 1))     # e.g., -32 for 6-bit
        
        print(f"Kernel Generator Configuration:")
        print(f"  Input channels: {in_channels}")
        print(f"  Output channels: {out_channels}")
        print(f"  Kernel size: {kernel_size}x{kernel_size}")
        print(f"  Bits per weight: {bits_per_weight}")
        print(f"  Value range: [{self.min_value}, {self.max_value}]")
        
    def generate_predefined_kernels(self) -> dict:
        """Generate some predefined useful kernels (edge detection, blur, etc.)"""
        kernels = {}
        
        # 3x3 kernels
        if self.kernel_size == 3:
            edge_detector = np.array([
                [-1, -1, -1],
                [-1,  8, -1],
                [-1, -1, -1]
            ])
            
            blur = np.array([
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]
            ]) // 4  # Scale down to fit in range
            
            sharpen = np.array([
                [ 0, -1,  0],
                [-1,  5, -1],
                [ 0, -1,  0]
            ])
            
            identity = np.array([
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]
            ])
            
            sobel_x = np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ])
            
            sobel_y = np.array([
                [-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1]
            ])
            
            kernels = {
                'edge_detector': edge_detector,
                'blur': blur,
                'sharpen': sharpen,
                'identity': identity,
                'sobel_x': sobel_x,
                'sobel_y': sobel_y,
            }
        else:
            # For other kernel sizes, generate simple patterns
            center = self.kernel_size // 2
            
            # Identity kernel
            identity = np.zeros((self.kernel_size, self.kernel_size), dtype=np.int8)
            identity[center, center] = 1
            
            # Simple edge detector
            edge = np.full((self.kernel_size, self.kernel_size), -1, dtype=np.int8)
            edge[center, center] = (self.kernel_size * self.kernel_size - 1)
            
            kernels = {
                'identity': identity,
                'edge_detector': edge,
            }
        
        return kernels
    
    def generate_random_kernels(self, count: int) -> dict:
        """Generate random kernels within the value range"""
        kernels = {}
        
        for i in range(count):
            kernel = np.random.randint(
                self.min_value, 
                self.max_value + 1, 
                size=(self.kernel_size, self.kernel_size),
                dtype=np.int8
            )
            kernels[f'random_{i}'] = kernel
            
        return kernels
    
    def assign_kernels_to_channels(self, kernel_dict: dict) -> np.ndarray:
        """Assign kernels to input/output channel combinations"""
        # Shape: [in_channels, out_channels, kernel_size, kernel_size]
        weights = np.zeros((self.in_channels, self.out_channels, 
                           self.kernel_size, self.kernel_size), dtype=np.int8)
        
        kernel_names = list(kernel_dict.keys())
        kernel_idx = 0
        
        print(f"\nKernel assignments:")
        for in_ch in range(self.in_channels):
            for out_ch in range(self.out_channels):
                kernel_name = kernel_names[kernel_idx % len(kernel_names)]
                weights[in_ch, out_ch] = kernel_dict[kernel_name]
                print(f"  Channel {in_ch} -> Channel {out_ch}: {kernel_name}")
                kernel_idx += 1
                
        return weights
    
    def clamp_weights(self, weights: np.ndarray) -> np.ndarray:
        """Clamp weights to the valid range"""
        return np.clip(weights, self.min_value, self.max_value)
    
    def weights_to_vivado_format(self, weights: np.ndarray) -> list:
        """Convert weights to Vivado BRAM format
        
        Address layout: in_channel * kernel_positions + position
        Data layout: packed output channels (LSB = channel 0)
        """
        vivado_data = []
        
        # Total BRAM words needed
        total_words = self.in_channels * self.total_positions
        
        for addr in range(total_words):
            # Decode address
            in_ch = addr // self.total_positions
            pos = addr % self.total_positions
            
            # Convert position to row, col
            row = pos // self.kernel_size
            col = pos % self.kernel_size
            
            # Pack all output channels for this position
            packed_value = 0
            for out_ch in range(self.out_channels):
                weight = weights[in_ch, out_ch, row, col]
                
                # Convert to unsigned representation for packing
                if weight < 0:
                    weight_unsigned = weight + (1 << self.bits_per_weight)
                else:
                    weight_unsigned = weight
                    
                # Pack into the word
                packed_value |= (weight_unsigned << (out_ch * self.bits_per_weight))
            
            vivado_data.append({
                'address': addr,
                'data': packed_value,
                'in_ch': in_ch,
                'pos': pos,
                'row': row,
                'col': col,
                'weights': [weights[in_ch, out_ch, row, col] for out_ch in range(self.out_channels)]
            })
            
        return vivado_data
    
    def save_vivado_file(self, vivado_data: list, filename: str):
        """Save Vivado BRAM initialization file"""
        # Calculate hex width
        total_bits = self.out_channels * self.bits_per_weight
        hex_width = (total_bits + 3) // 4
        
        with open(filename, 'w') as f:
            f.write(f"// Kernel weights for Vivado BRAM initialization\n")
            f.write(f"// Input channels: {self.in_channels}\n")
            f.write(f"// Output channels: {self.out_channels}\n")
            f.write(f"// Kernel size: {self.kernel_size}x{self.kernel_size}\n")
            f.write(f"// Bits per weight: {self.bits_per_weight}\n")
            f.write(f"// Total bits per word: {total_bits}\n")
            f.write(f"// Data format: output_ch0[{self.bits_per_weight-1}:0], ")
            f.write(f"output_ch1[{2*self.bits_per_weight-1}:{self.bits_per_weight}], ...\n")
            f.write(f"// Address format: in_channel * {self.total_positions} + position\n\n")
            
            for item in vivado_data:
                f.write(f"@{item['address']:03X} {item['data']:0{hex_width}X}  ")
                f.write(f"// In_ch{item['in_ch']} pos({item['row']},{item['col']}) weights: {item['weights']}\n")
                
        print(f"Vivado file saved: {filename}")
    
    def save_python_file(self, weights: np.ndarray, filename: str):
        """Save Python-compatible file for mockup scripts"""
        
        # Convert to standard Python types for JSON serialization
        weights_list = weights.astype(int).tolist()
        
        data = {
            'config': {
                'in_channels': self.in_channels,
                'out_channels': self.out_channels,
                'kernel_size': self.kernel_size,
                'bits_per_weight': self.bits_per_weight,
                'value_range': [self.min_value, self.max_value]
            },
            'weights': weights_list,
            'shape': list(weights.shape),
            'description': 'Kernel weights in format [in_channels][out_channels][row][col]'
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Python file saved: {filename}")
    
    def save_numpy_file(self, weights: np.ndarray, filename: str):
        """Save as numpy file for easy loading"""
        np.save(filename, weights)
        print(f"Numpy file saved: {filename}")
    
    def print_summary(self, weights: np.ndarray):
        """Print a summary of the generated weights"""
        print(f"\nWeight Summary:")
        print(f"  Shape: {weights.shape}")
        print(f"  Min value: {weights.min()}")
        print(f"  Max value: {weights.max()}")
        print(f"  Mean: {weights.mean():.2f}")
        print(f"  Non-zero weights: {np.count_nonzero(weights)}/{weights.size}")
        
        # Show first kernel as example
        if weights.size > 0:
            print(f"\nExample kernel (in_ch=0, out_ch=0):")
            print(weights[0, 0])

def main():
    parser = argparse.ArgumentParser(description='Generate kernel weights for SNN convolution')
    parser.add_argument('in_channels', type=int, help='Number of input channels')
    parser.add_argument('out_channels', type=int, help='Number of output channels')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size (default: 3)')
    parser.add_argument('--bits_per_weight', type=int, default=6, help='Bits per weight (default: 6)')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory')
    parser.add_argument('--prefix', type=str, default='kernel', help='Output file prefix')
    parser.add_argument('--random_kernels', type=int, default=0, help='Number of additional random kernels')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create generator
    generator = KernelGenerator(
        args.in_channels, 
        args.out_channels, 
        args.kernel_size,
        args.bits_per_weight
    )
    
    # Generate kernels
    predefined_kernels = generator.generate_predefined_kernels()
    
    if args.random_kernels > 0:
        random_kernels = generator.generate_random_kernels(args.random_kernels)
        all_kernels = {**predefined_kernels, **random_kernels}
    else:
        all_kernels = predefined_kernels
    
    print(f"\nAvailable kernels: {list(all_kernels.keys())}")
    
    # Assign kernels to channels
    weights = generator.assign_kernels_to_channels(all_kernels)
    weights = generator.clamp_weights(weights)
    
    # Generate output files
    vivado_data = generator.weights_to_vivado_format(weights)
    
    vivado_file = output_dir / f"{args.prefix}_weights.mem"
    python_file = output_dir / f"{args.prefix}_weights.json"
    numpy_file = output_dir / f"{args.prefix}_weights.npy"
    
    generator.save_vivado_file(vivado_data, vivado_file)
    generator.save_python_file(weights, python_file)
    generator.save_numpy_file(weights, numpy_file)
    
    # Print summary
    generator.print_summary(weights)
    
    print(f"\n✓ Generated kernel weights:")
    print(f"  Vivado BRAM file: {vivado_file}")
    print(f"  Python JSON file: {python_file}")
    print(f"  Numpy file: {numpy_file}")

if __name__ == "__main__":
    main()