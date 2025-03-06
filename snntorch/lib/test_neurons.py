from Neurons import LIFNeuron
import torch

num_synapses = 5

neuron = LIFNeuron(num_synapses)

input_signal = torch.tensor([1, 0, 1, 1, 0], dtype=torch.int8)

output_spike = neuron(input_signal)

# Print neuron state
print("Spike:", output_spike.item())
print("Membrane Potential:", neuron.membrane_potential.item())
print("Threshold:", neuron.threshold.item())
print("Reset Potential:", neuron.membrane_potential_reset.item())
print("Leakage Strength:", neuron.leakage_strength.item())