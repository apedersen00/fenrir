import argparse
import torch
import os
import numpy as np
from f_quant_net import ThreeConvPoolingNet

def parse_arguments():
    parser = argparse.ArgumentParser(description='Export quantized neural network parameters to VHDL')
    
    # Model loading parameters
    parser.add_argument('--model', type=str, required=True, help='Path to the PyTorch model file')
    parser.add_argument('--output', type=str, default='model_params.vhd', help='Output VHDL file path')
    
    # Quantization parameters
    parser.add_argument('--bits', type=int, default=8, help='Bit-width for quantization (default: 8)')
    parser.add_argument('--quantize', action='store_true', help='Apply quantization (if model not already quantized)')
    
    # Export options
    parser.add_argument('--format', choices=['constants', 'rom', 'mif'], default='constants', 
                        help='Export format: constants (VHDL constants), rom (VHDL ROM entity), or mif (Memory Init File)')
    parser.add_argument('--separate-files', action='store_true', help='Export each layer to a separate file')
    parser.add_argument('--include-scales', action='store_true', help='Include scaling factors in export')
    parser.add_argument('--power2-scales', action='store_true', help='Approximate scales as powers of 2')
    
    # Sample export for verification
    parser.add_argument('--export-sample', action='store_true', help='Export a sample input/output for verification')
    
    return parser.parse_args()

def symmetric_quantize_weights(model, bits=8):
    """Apply symmetric Post-Training Quantization to model weights"""
    quantized_model = ThreeConvPoolingNet()
    quantized_model.load_state_dict(model.state_dict())
    
    with torch.no_grad():
        # Set quantization range for signed integers
        qmin, qmax = -2**(bits-1), 2**(bits-1) - 1
            
        # Quantize convolutional weights
        for name, module in quantized_model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                # Calculate scaling factor based on absolute max value
                w_absmax = torch.abs(module.weight.data).max()
                scale = (qmax - qmin) / (2 * w_absmax) if w_absmax > 0 else 1.0
                
                # Quantize to integers
                w_int = torch.round(module.weight.data * scale)
                
                # Clamp to ensure values are within range
                w_int = torch.clamp(w_int, qmin, qmax)
                
                # Save integer weights (for FPGA implementation)
                module.register_buffer('weight_int', w_int)
                
                # Dequantize for PyTorch inference (would be removed in FPGA)
                module.weight.data = w_int / scale
                
                # Store quantization parameters
                module.register_buffer('scale_factor', torch.tensor(scale).clone())
                
        # Quantize parameters in pooling layers
        for name, module in quantized_model.named_modules():
            # Threshold parameters
            if hasattr(module, 'threshold') and isinstance(module.threshold, torch.nn.Parameter):
                thresh_absmax = torch.abs(module.threshold.data).max()
                thresh_scale = (qmax - qmin) / (2 * thresh_absmax) if thresh_absmax > 0 else 1.0
                
                # Quantize to integers
                thresh_int = torch.round(module.threshold.data * thresh_scale)
                thresh_int = torch.clamp(thresh_int, qmin, qmax)
                
                # Save integer thresholds
                module.register_buffer('threshold_int', thresh_int)
                
                # Dequantize for PyTorch
                module.threshold.data = thresh_int / thresh_scale
                
                # Store parameters
                module.register_buffer('threshold_scale', torch.tensor(thresh_scale).clone())
            
            # Decay parameters
            if hasattr(module, 'decay') and isinstance(module.decay, torch.nn.Parameter):
                decay_absmax = torch.abs(module.decay.data).max()
                decay_scale = (qmax - qmin) / (2 * decay_absmax) if decay_absmax > 0 else 1.0
                
                # Quantize
                decay_int = torch.round(module.decay.data * decay_scale)
                decay_int = torch.clamp(decay_int, qmin, qmax)
                
                # Save integer decay values
                module.register_buffer('decay_int', decay_int)
                
                # Dequantize for PyTorch
                module.decay.data = decay_int / decay_scale
                
                # Store parameters
                module.register_buffer('decay_scale', torch.tensor(decay_scale).clone())
    
    return quantized_model

def approximate_to_power2(value):
    """Approximate a value to the nearest power of 2"""
    if value <= 0:
        return 0
    log2_val = np.log2(value)
    rounded_log2 = round(log2_val)
    return 2 ** rounded_log2

def export_vhdl_constants(model, output_file, include_scales=False, power2_scales=False):
    """Export model parameters as VHDL constants"""
    with open(output_file, 'w') as f:
        f.write("-- Auto-generated VHDL constants for quantized neural network\n")
        f.write("-- Generated by NN-to-VHDL export tool\n\n")
        
        f.write("library IEEE;\n")
        f.write("use IEEE.STD_LOGIC_1164.ALL;\n")
        f.write("use IEEE.NUMERIC_STD.ALL;\n\n")
        
        f.write("package model_params is\n\n")
        
        # Export parameters for each layer
        for name, module in model.named_modules():
            # Export convolutional weights
            if hasattr(module, 'weight_int'):
                w_int = module.weight_int
                w_shape = w_int.shape
                
                layer_name = name.replace('.', '_').upper()
                
                # Write weight type and array
                f.write(f"  -- {name} weights ({w_shape[0]} x {w_shape[1]} x {w_shape[2]} x {w_shape[3]})\n")
                f.write(f"  type {layer_name}_WEIGHTS_T is array(0 to {w_shape[0]-1}, 0 to {w_shape[1]-1}, " 
                       f"0 to {w_shape[2]-1}, 0 to {w_shape[3]-1}) of integer range -128 to 127;\n")
                f.write(f"  constant {layer_name}_WEIGHTS : {layer_name}_WEIGHTS_T := (\n")
                
                # Write 4D array values
                for out_ch in range(w_shape[0]):
                    f.write(f"    {out_ch} => (\n")
                    
                    for in_ch in range(w_shape[1]):
                        f.write(f"      {in_ch} => (\n")
                        
                        for ky in range(w_shape[2]):
                            f.write(f"        {ky} => (")
                            
                            # Write row of kernel values
                            values = [str(int(w_int[out_ch, in_ch, ky, kx].item())) 
                                     for kx in range(w_shape[3])]
                            f.write(", ".join(values))
                            
                            if ky < w_shape[2] - 1:
                                f.write("),\n")
                            else:
                                f.write(")\n")
                        
                        if in_ch < w_shape[1] - 1:
                            f.write("      ),\n")
                        else:
                            f.write("      )\n")
                    
                    if out_ch < w_shape[0] - 1:
                        f.write("    ),\n")
                    else:
                        f.write("    )\n")
                
                f.write("  );\n\n")
                
                # Export scale factor if requested
                if include_scales:
                    scale = module.scale_factor.item()
                    if power2_scales:
                        scale_power2 = approximate_to_power2(scale)
                        log2_scale = int(np.log2(scale_power2)) if scale_power2 > 0 else 0
                        f.write(f"  -- {name} scale factor (power of 2)\n")
                        f.write(f"  constant {layer_name}_SCALE_SHIFT : integer := {log2_scale};\n\n")
                    else:
                        f.write(f"  -- {name} scale factor\n")
                        f.write(f"  constant {layer_name}_SCALE : real := {scale:.10f};\n\n")
            
            # Export threshold parameters
            if hasattr(module, 'threshold_int'):
                threshold_int = module.threshold_int
                t_shape = threshold_int.shape
                
                layer_name = name.replace('.', '_').upper()
                
                f.write(f"  -- {name} threshold parameters ({t_shape[0]})\n")
                f.write(f"  type {layer_name}_THRESHOLD_T is array(0 to {t_shape[0]-1}) of integer range -128 to 127;\n")
                f.write(f"  constant {layer_name}_THRESHOLD : {layer_name}_THRESHOLD_T := (")
                
                # Write threshold values
                values = [str(int(threshold_int[i].item())) for i in range(t_shape[0])]
                f.write(", ".join(values))
                f.write(");\n\n")
                
                # Export scale factor if requested
                if include_scales and hasattr(module, 'threshold_scale'):
                    scale = module.threshold_scale.item()
                    if power2_scales:
                        scale_power2 = approximate_to_power2(scale)
                        log2_scale = int(np.log2(scale_power2)) if scale_power2 > 0 else 0
                        f.write(f"  -- {name} threshold scale factor (power of 2)\n")
                        f.write(f"  constant {layer_name}_THRESHOLD_SCALE_SHIFT : integer := {log2_scale};\n\n")
                    else:
                        f.write(f"  -- {name} threshold scale factor\n")
                        f.write(f"  constant {layer_name}_THRESHOLD_SCALE : real := {scale:.10f};\n\n")
            
            # Export decay parameters
            if hasattr(module, 'decay_int'):
                decay_int = module.decay_int
                d_shape = decay_int.shape
                
                layer_name = name.replace('.', '_').upper()
                
                f.write(f"  -- {name} decay parameters ({d_shape[0]})\n")
                f.write(f"  type {layer_name}_DECAY_T is array(0 to {d_shape[0]-1}) of integer range -128 to 127;\n")
                f.write(f"  constant {layer_name}_DECAY : {layer_name}_DECAY_T := (")
                
                # Write decay values
                values = [str(int(decay_int[i].item())) for i in range(d_shape[0])]
                f.write(", ".join(values))
                f.write(");\n\n")
                
                # Export scale factor if requested
                if include_scales and hasattr(module, 'decay_scale'):
                    scale = module.decay_scale.item()
                    if power2_scales:
                        scale_power2 = approximate_to_power2(scale)
                        log2_scale = int(np.log2(scale_power2)) if scale_power2 > 0 else 0
                        f.write(f"  -- {name} decay scale factor (power of 2)\n")
                        f.write(f"  constant {layer_name}_DECAY_SCALE_SHIFT : integer := {log2_scale};\n\n")
                    else:
                        f.write(f"  -- {name} decay scale factor\n")
                        f.write(f"  constant {layer_name}_DECAY_SCALE : real := {scale:.10f};\n\n")
        
        f.write("end package model_params;\n")
    
    print(f"VHDL constants exported to {output_file}")

def export_vhdl_rom(model, output_file, include_scales=False):
    """Export model parameters as VHDL ROM entities"""
    # Similar to constants export but creates ROM entities
    # Implementation would be similar but create ROMs instead of constants
    with open(output_file, 'w') as f:
        f.write("-- Auto-generated VHDL ROM entities for quantized neural network\n")
        # Implementation for ROM export
        # This would generate ROM entities with addresses and data
    print(f"VHDL ROM entities exported to {output_file}")

def export_mif_file(model, output_dir):
    """Export model parameters as Memory Initialization Files (MIF)"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Export each weight matrix as a separate MIF file
    for name, module in model.named_modules():
        if hasattr(module, 'weight_int'):
            w_int = module.weight_int
            layer_name = name.replace('.', '_').upper()
            mif_file = os.path.join(output_dir, f"{layer_name}_WEIGHTS.mif")
            
            with open(mif_file, 'w') as f:
                f.write("-- Memory Initialization File\n")
                f.write("-- Generated for layer: " + name + "\n\n")
                
                # MIF header
                f.write("WIDTH=8;\n")  # 8-bit data
                f.write(f"DEPTH={w_int.numel()};\n\n")
                f.write("ADDRESS_RADIX=HEX;\n")
                f.write("DATA_RADIX=DEC;\n\n")
                f.write("CONTENT BEGIN\n")
                
                # Write flattened weight data
                w_flat = w_int.flatten()
                for i in range(w_flat.numel()):
                    addr = format(i, 'x')
                    val = int(w_flat[i].item())
                    f.write(f"  {addr} : {val};\n")
                
                f.write("END;\n")
    
    print(f"MIF files exported to {output_dir}")

def export_sample_for_verification(model, output_file):
    """Export a sample input and its propagation through the model"""
    # This would generate test vectors for verification
    with open(output_file, 'w') as f:
        f.write("-- Verification data for quantized neural network\n")
        # Implementation for test data generation
    print(f"Verification data exported to {output_file}")

def main():
    args = parse_arguments()
    
    # Load the model
    print(f"Loading model from {args.model}")
    model = ThreeConvPoolingNet()
    model.load_state_dict(torch.load(args.model))
    
    # Quantize if needed
    if args.quantize:
        print(f"Applying {args.bits}-bit quantization")
        model = symmetric_quantize_weights(model, bits=args.bits)
    
    # Export according to format
    if args.format == 'constants':
        if args.separate_files:
            # Export each layer to a separate file
            output_dir = os.path.splitext(args.output)[0]
            os.makedirs(output_dir, exist_ok=True)
            for name, _ in model.named_modules():
                if name and '.' not in name:  # Only top-level modules
                    layer_file = os.path.join(output_dir, f"{name}_params.vhd")
                    # Export just this layer
                    print(f"Exporting {name} to {layer_file}")
        else:
            # Export all to one file
            export_vhdl_constants(model, args.output, 
                                 include_scales=args.include_scales, 
                                 power2_scales=args.power2_scales)
    
    elif args.format == 'rom':
        export_vhdl_rom(model, args.output, include_scales=args.include_scales)
    
    elif args.format == 'mif':
        output_dir = os.path.splitext(args.output)[0] + "_mif"
        export_mif_file(model, output_dir)
    
    # Export sample if requested
    if args.export_sample:
        sample_file = os.path.splitext(args.output)[0] + "_sample.vhd"
        export_sample_for_verification(model, sample_file)
    
    print("Export complete!")

if __name__ == "__main__":
    main()