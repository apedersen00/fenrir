import torch
import torch.nn as nn

class LIFNeuron(nn.Module):
    def __init__(
            self, 
            num_synapses: int,
            threshold_bits: int = 12,
            leakage_strength_bits: int = 6,
            membrane_potential_reset_bits: int = 2,
            weight_bits: int = 2,
            membrane_potential_bits: int = 12
            ):
        
        super().__init__()

        self.num_synapses               = num_synapses
        self.threshold                  = nn.Parameter(torch.rand(1, dtype=torch.float32))  
        self.leakage_strength           = nn.Parameter(torch.rand(1, dtype=torch.float32))  
        self.membrane_potential_reset   = nn.Parameter(torch.rand(1, dtype=torch.float32))  
        self.weights                    = nn.Parameter(torch.rand(num_synapses, dtype=torch.float32))  
        self.membrane_potential         = torch.zeros(1, dtype=torch.int16)  # Range -2048 to 2047

        self.w_threshold                = threshold_bits
        self.w_leakage_strength         = leakage_strength_bits
        self.w_membrane_potential_reset = membrane_potential_reset_bits
        self.w_weights                  = weight_bits
        self.w_membrane_potential       = membrane_potential_bits

    def quantize(
            self, 
            x: torch.Tensor,
            bits: int, 
            signed:bool =True
            ):
        
        qmin    = -(2**(bits-1)) if signed else 0
        qmax    = (2**(bits-1)) - 1 if signed else (2**bits) - 1
        scale   = qmax - qmin
        return torch.round(x * scale).clamp(qmin, qmax) / scale

    def forward(self, inputs):
        
        assert len(inputs) == self.num_synapses, "Mismatch in input size"

        # Apply quantization to enforce bit constraints
        leakage_strength            = self.quantize(self.leakage_strength, self.w_leakage_strength)
        threshold                   = self.quantize(self.threshold, self.w_threshold)
        membrane_potential_reset    = self.quantize(self.membrane_potential_reset, self.w_membrane_potential_reset)
        weights                     = self.quantize(self.weights, self.w_weights, signed=False)

        for i in range(self.num_synapses):
            self.membrane_potential += weights[i].to(torch.int16)

        self.membrane_potential = torch.clamp(self.membrane_potential, -1*2**(self.w_membrane_potential - 1), 2**(self.w_membrane_potential - 1) - 1)

        spike = self.membrane_potential >= threshold
        if spike:
            self.membrane_potential = membrane_potential_reset.clone().detach()
        else:
            self.membrane_potential = self.membrane_potential - leakage_strength

        return spike
