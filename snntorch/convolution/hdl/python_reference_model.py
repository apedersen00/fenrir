#!/usr/bin/env python3
"""
SNN Processing Reference Model
Generates test vectors and expected outputs for VHDL verification
"""

import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration matching VHDL generics"""
    IMG_WIDTH: int = 16
    IMG_HEIGHT: int = 16
    NEURON_BIT_WIDTH: int = 9
    KERNEL_SIZE: int = 3
    POOL_SIZE: int = 2
    CHANNELS_OUT: int = 4
    BITS_PER_WEIGHT: int = 9
    RESET_VALUE: int = 0
    
    # Pooling parameters (matching VHDL)
    def get_channel_thresholds(self):
        return [80 + ch * 20 for ch in range(self.CHANNELS_OUT)]
    
    def get_channel_decay(self):
        return [2 for ch in range(self.CHANNELS_OUT)]

@dataclass 
class Event:
    """Single input event"""
    x: int
    y: int
    timestep: int = 0

class SNNReferenceModel:
    """Reference implementation of SNN processing pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.membrane_potentials = np.zeros((config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS_OUT), dtype=np.int32)
        self.kernel_weights = self._init_kernel_weights()
        self.thresholds = config.get_channel_thresholds()
        self.decay_values = config.get_channel_decay()
        
        # Results storage
        self.membrane_before_pooling = None
        self.membrane_after_pooling = None
        self.spike_events = []
        
    def _init_kernel_weights(self):
        """Initialize kernel weights matching VHDL implementation"""
        weights = np.zeros((self.config.KERNEL_SIZE, self.config.KERNEL_SIZE, self.config.CHANNELS_OUT), dtype=np.int32)
        
        for ky in range(self.config.KERNEL_SIZE):
            for kx in range(self.config.KERNEL_SIZE):
                pos = ky * self.config.KERNEL_SIZE + kx
                for ch in range(self.config.CHANNELS_OUT):
                    # Match VHDL weight initialization
                    if (pos + ch) % 3 == 0:
                        weights[ky, kx, ch] = -(pos + ch + 1)  # Negative
                    else:
                        weights[ky, kx, ch] = (pos + ch + 1)   # Positive
                        
        return weights
    
    def _get_kernel_coords(self, center_x: int, center_y: int) -> List[Tuple[int, int]]:
        """Get valid kernel coordinates around center point"""
        coords = []
        half_kernel = self.config.KERNEL_SIZE // 2
        
        for ky in range(self.config.KERNEL_SIZE):
            for kx in range(self.config.KERNEL_SIZE):
                offset_x = kx - half_kernel
                offset_y = ky - half_kernel
                coord_x = center_x + offset_x
                coord_y = center_y + offset_y
                
                # Check bounds
                if (0 <= coord_x < self.config.IMG_WIDTH and 
                    0 <= coord_y < self.config.IMG_HEIGHT):
                    coords.append((coord_x, coord_y, kx, ky))
                    
        return coords
    
    def _saturate(self, value: int, bits: int) -> int:
        """Saturate value to signed bit width"""
        max_val = (1 << (bits - 1)) - 1
        min_val = -(1 << (bits - 1))
        return max(min_val, min(max_val, value))
    
    def process_event(self, event: Event):
        """Process single event through convolution"""
        coords = self._get_kernel_coords(event.x, event.y)
        
        for coord_x, coord_y, kx, ky in coords:
            # Get current membrane potential
            current_membrane = self.membrane_potentials[coord_y, coord_x, :]
            
            # Apply convolution (add kernel weights)
            for ch in range(self.config.CHANNELS_OUT):
                weight = self.kernel_weights[ky, kx, ch]
                new_value = current_membrane[ch] + weight
                
                # Saturate to bit width
                self.membrane_potentials[coord_y, coord_x, ch] = self._saturate(
                    new_value, self.config.NEURON_BIT_WIDTH
                )
    
    def _apply_decay_and_check_spikes(self, membrane_val: int, threshold: int, decay: int) -> Tuple[int, bool]:
        """Apply decay and check for spike, return (new_value, spiked)"""
        # Apply decay
        decayed = max(0, membrane_val - decay)
        
        # Check threshold
        if decayed >= threshold:
            return self.config.RESET_VALUE, True  # Spike and reset
        else:
            return decayed, False  # No spike
    
    def process_pooling(self):
        """Process pooling with decay and spike generation"""
        # Store membrane state before pooling
        self.membrane_before_pooling = self.membrane_potentials.copy()
        
        # Initialize spike events list
        self.spike_events = []
        
        # Process each pooling window
        windows_x = self.config.IMG_WIDTH // self.config.POOL_SIZE
        windows_y = self.config.IMG_HEIGHT // self.config.POOL_SIZE
        
        for win_y in range(windows_y):
            for win_x in range(windows_x):
                # Accumulate membrane potentials in this window
                window_accumulator = np.zeros(self.config.CHANNELS_OUT, dtype=np.int32)
                
                # Process each pixel in window
                for py in range(self.config.POOL_SIZE):
                    for px in range(self.config.POOL_SIZE):
                        pixel_x = win_x * self.config.POOL_SIZE + px
                        pixel_y = win_y * self.config.POOL_SIZE + py
                        
                        # Process each channel
                        for ch in range(self.config.CHANNELS_OUT):
                            membrane_val = self.membrane_potentials[pixel_y, pixel_x, ch]
                            
                            # Apply decay and check for spike
                            new_val, spiked = self._apply_decay_and_check_spikes(
                                membrane_val, self.thresholds[ch], self.decay_values[ch]
                            )
                            
                            # Update membrane potential
                            self.membrane_potentials[pixel_y, pixel_x, ch] = new_val
                            
                            # If no spike, add to accumulator
                            if not spiked:
                                window_accumulator[ch] += new_val
                
                # Generate spike vector for this window
                spike_vector = np.zeros(self.config.CHANNELS_OUT, dtype=bool)
                for ch in range(self.config.CHANNELS_OUT):
                    if window_accumulator[ch] >= self.thresholds[ch]:
                        spike_vector[ch] = True
                
                # Add spike event if any spikes occurred
                if np.any(spike_vector):
                    self.spike_events.append({
                        'x': win_x,
                        'y': win_y, 
                        'spikes': spike_vector.tolist()
                    })
        
        # Store membrane state after pooling
        self.membrane_after_pooling = self.membrane_potentials.copy()

def generate_test_events(config: Config, num_events: int = 20, seed: int = 42) -> List[Event]:
    """Generate pseudo-random test events"""
    np.random.seed(seed)
    events = []
    
    # Generate some clustered events (more realistic for neural processing)
    cluster_centers = [
        (4, 4), (12, 4), (4, 12), (12, 12),  # Corners
        (8, 8),                               # Center
        (2, 8), (14, 8)                      # Sides
    ]
    
    for i in range(num_events):
        if i < len(cluster_centers):
            # Use predefined centers for first events
            center_x, center_y = cluster_centers[i]
        else:
            # Random locations for remaining events
            center_x = np.random.randint(2, config.IMG_WIDTH - 2)
            center_y = np.random.randint(2, config.IMG_HEIGHT - 2)
        
        # Add some noise around center
        noise_x = np.random.randint(-1, 2)
        noise_y = np.random.randint(-1, 2)
        
        x = max(0, min(config.IMG_WIDTH - 1, center_x + noise_x))
        y = max(0, min(config.IMG_HEIGHT - 1, center_y + noise_y))
        
        events.append(Event(x=x, y=y, timestep=i))
    
    return events

def save_test_files(events: List[Event], model: SNNReferenceModel, output_dir: str = "test_vectors"):
    """Save test vectors and expected outputs to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save input events (for VHDL testbench)
    with open(f"{output_dir}/input_events.txt", "w") as f:
        f.write(f"# Input events: x y timestep\n")
        for event in events:
            # Format: 8-bit x, 8-bit y coordinates as hex
            f.write(f"{event.x:02X}{event.y:02X}\n")
    
    # Save expected membrane potentials before pooling
    with open(f"{output_dir}/membrane_before_pooling.txt", "w") as f:
        f.write(f"# Membrane potentials before pooling: addr channel0 channel1 channel2 channel3\n")
        for y in range(model.config.IMG_HEIGHT):
            for x in range(model.config.IMG_WIDTH):
                addr = y * model.config.IMG_WIDTH + x
                values = model.membrane_before_pooling[y, x, :]
                f.write(f"{addr:04X}")
                for val in values:
                    # Convert signed to unsigned for file format
                    unsigned_val = val & ((1 << model.config.NEURON_BIT_WIDTH) - 1)
                    f.write(f" {unsigned_val:03X}")
                f.write("\n")
    
    # Save expected membrane potentials after pooling 
    with open(f"{output_dir}/membrane_after_pooling.txt", "w") as f:
        f.write(f"# Membrane potentials after pooling: addr channel0 channel1 channel2 channel3\n")
        for y in range(model.config.IMG_HEIGHT):
            for x in range(model.config.IMG_WIDTH):
                addr = y * model.config.IMG_WIDTH + x
                values = model.membrane_after_pooling[y, x, :]
                f.write(f"{addr:04X}")
                for val in values:
                    unsigned_val = val & ((1 << model.config.NEURON_BIT_WIDTH) - 1)
                    f.write(f" {unsigned_val:03X}")
                f.write("\n")
    
    # Save expected spike events
    with open(f"{output_dir}/expected_spikes.txt", "w") as f:
        f.write(f"# Expected spike events: x y spike_vector\n")
        for spike_event in model.spike_events:
            spike_bits = 0
            for i, spike in enumerate(spike_event['spikes']):
                if spike:
                    spike_bits |= (1 << i)
            f.write(f"{spike_event['x']:02X}{spike_event['y']:02X} {spike_bits:02X}\n")
    
    # Save configuration for reference
    config_dict = {
        'IMG_WIDTH': model.config.IMG_WIDTH,
        'IMG_HEIGHT': model.config.IMG_HEIGHT,
        'NEURON_BIT_WIDTH': model.config.NEURON_BIT_WIDTH,
        'KERNEL_SIZE': model.config.KERNEL_SIZE,
        'POOL_SIZE': model.config.POOL_SIZE,
        'CHANNELS_OUT': model.config.CHANNELS_OUT,
        'thresholds': model.thresholds,
        'decay_values': model.decay_values,
        'num_events': len(events),
        'num_spike_events': len(model.spike_events)
    }
    
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Test vectors saved to {output_dir}/")
    print(f"  Input events: {len(events)}")
    print(f"  Expected spike events: {len(model.spike_events)}")
    print(f"  Configuration: {output_dir}/config.json")

def main():
    """Generate test vectors and run reference model"""
    config = Config()
    
    print("SNN Reference Model - Test Vector Generation")
    print(f"Image size: {config.IMG_WIDTH}x{config.IMG_HEIGHT}")
    print(f"Channels: {config.CHANNELS_OUT}, Kernel: {config.KERNEL_SIZE}x{config.KERNEL_SIZE}")
    print(f"Thresholds: {config.get_channel_thresholds()}")
    print(f"Decay values: {config.get_channel_decay()}")
    
    # Generate test events
    events = generate_test_events(config, num_events=16)
    print(f"\nGenerated {len(events)} test events")
    
    # Process through reference model
    model = SNNReferenceModel(config)
    
    print("\nProcessing events through convolution...")
    for i, event in enumerate(events):
        model.process_event(event)
        if i % 5 == 4:
            print(f"  Processed {i+1}/{len(events)} events")
    
    print("Processing pooling...")
    model.process_pooling()
    
    print(f"Generated {len(model.spike_events)} spike events")
    
    # Save test files
    save_test_files(events, model)
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"  Max membrane potential: {np.max(model.membrane_before_pooling)}")
    print(f"  Min membrane potential: {np.min(model.membrane_before_pooling)}")
    print(f"  Average membrane potential: {np.mean(model.membrane_before_pooling):.2f}")
    print(f"  Spike events generated: {len(model.spike_events)}")
    
    if model.spike_events:
        total_spikes = sum(sum(event['spikes']) for event in model.spike_events)
        print(f"  Total individual spikes: {total_spikes}")

if __name__ == "__main__":
    main()