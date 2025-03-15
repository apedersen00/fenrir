from hwcomponents import *
from Controller import *

settings = VortexSetings(
    amount_neurons=10,
    amount_synapses=10,
    amount_samples=3,
    neuron=Neurons.LIF()
)

word_sizes = {
    "bram_neuron": 32,
    "bram_synapse": 32,
    "param_leak_str": 5,
    "param_threshold": 5,
    "param_reset": 5,
    "state_core": 5,
    "synapses_width": 8
}

Export(settings, word_sizes)