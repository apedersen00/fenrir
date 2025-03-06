import torch
import torch.nn as nn

class SurrogateStepFunction(torch.autograd.Function):
    """ Custom fast sigmoid used for backpropagation (NOT in forward pass) """

    @staticmethod
    def forward(ctx, x):
        """ Forward pass: Hard thresholding (step function) """
        ctx.save_for_backward(x)
        return (x >= 0).float()  # Binary output (0 or 1)

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass: Fast sigmoid as surrogate gradient """
        x, = ctx.saved_tensors
        grad_input = grad_output * (1 / (1 + torch.abs(x))) ** 2  # Fast sigmoid derivative
        return grad_input

def fast_sigmoid_grad(x):
    """ Use fast sigmoid for backpropagation """
    return SurrogateStepFunction.apply(x)

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
        self.membrane_potential         = torch.zeros(1, dtype=torch.float32, requires_grad=False)  

        self.w_threshold                = threshold_bits
        self.w_leakage_strength         = leakage_strength_bits
        self.w_membrane_potential_reset = membrane_potential_reset_bits
        self.w_weights                  = weight_bits
        self.w_membrane_potential       = membrane_potential_bits

    def quantize(self, x, bits, signed=True):
        """Simulate Fixed-Point Quantization"""
        qmin = -(2**(bits-1)) if signed else 0
        qmax = (2**(bits-1)) - 1 if signed else (2**bits) - 1
        scale = qmax - qmin
        return torch.round(x * scale).clamp(qmin, qmax) / scale  # Fake quantization

    def forward(self, inputs):
        """
        Inputs: List of synaptic activations (0 or 1)
        Outputs: Spike (Binary 0 or 1)
        """
        assert len(inputs) == self.weights.shape[0], "Mismatch in input size"

        # Apply quantization to enforce bit constraints
        leakage_strength = self.quantize(self.leakage_strength, self.w_leakage_strength)
        threshold = self.quantize(self.threshold, self.w_threshold)
        membrane_potential_reset = self.quantize(self.membrane_potential_reset, self.w_membrane_potential_reset)
        weights = self.quantize(self.weights, self.w_weights, signed=False)

        # Apply leakage decay
        self.membrane_potential -= leakage_strength

        # Iterate over each synapse
        for i in range(self.weights.shape[0]):
            self.membrane_potential += inputs[i] * weights[i]

        # Clamp membrane potential
        self.membrane_potential = torch.clamp(
            self.membrane_potential, 
            -1 * 2**(self.w_membrane_potential - 1), 
            2**(self.w_membrane_potential - 1) - 1
        )

        # Hard threshold for spike generation (NO surrogate function in forward pass)
        spike = (self.membrane_potential >= threshold).float()

        # Reset membrane potential if a spike occurs
        self.membrane_potential = torch.where(spike > 0, membrane_potential_reset, self.membrane_potential)

        return spike

    def compute_surrogate_loss(self, inputs, target_spikes):
        """
        Computes surrogate loss for training by applying fast sigmoid gradient to weight updates.
        """
        # Compute membrane potential update (like forward pass but differentiable)
        potential = -self.leakage_strength
        for i in range(self.weights.shape[0]):
            potential += inputs[i] * self.weights[i]

        # Compute surrogate gradient-based spike output (for loss)
        surrogate_spike = fast_sigmoid_grad(potential - self.threshold)

        # Loss function (Binary Cross Entropy for spike outputs)
        loss_function = nn.BCELoss()
        loss = loss_function(surrogate_spike, target_spikes)

        return loss

# ======== Training Setup ========
num_synapses = 5
neuron = LIFNeuron(num_synapses)

# Loss function (Binary Cross Entropy for spike outputs)
optimizer = torch.optim.Adam(neuron.parameters(), lr=0.01)

# Training Data (Random binary input and desired spike output)
inputs = torch.randint(0, 2, (10, num_synapses), dtype=torch.float32)  # 10 samples
targets = torch.randint(0, 2, (10, 1), dtype=torch.float32)  # 10 desired spike outputs

# Training loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()  # Reset gradients
    
    total_loss = 0
    for i in range(len(inputs)):
        loss = neuron.compute_surrogate_loss(inputs[i], targets[i])  # Compute loss using surrogate function
        loss.backward()  # Backpropagation
        total_loss += loss.item()

    optimizer.step()  # Update weights
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(inputs)}")

print("Training Complete!")
