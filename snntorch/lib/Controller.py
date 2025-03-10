import numpy as np
from hwcomponents import *
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

@dataclass
class VortexSetings:
    """
    Helper class to quickly set up a simulation
    """
    amount_neurons: int
    amount_synapses: int
    amount_samples: int
    
    neuron: Union[Neurons.LIF, any]

    InputData: Optional[DataStructures.InputData] = None

    def __post_init__(self):
        
        if self.InputData is None:
            self.InputData = DataStructures.InputData()
            for _ in range(self.amount_samples):
                self.InputData.add_sample(
                    DataStructures.InputSample(
                        length=self.amount_neurons,
                        randomize_fun=self.randSample
                    )
                )

    def randSample(self, length: int):
        return np.array([
            Utils.DiscreteNORM(.1, .5, -1, 1) for _ in range(length)
        ])



class VortexOne:

    def __init__(
            self,
            settings: Union[VortexSetings, dict] #for now we will only set it up for settings class, but for flexibility we might add dict later
    ):
        if isinstance(settings, VortexSetings):
            self.settings = settings
        else:
            raise ValueError("Settings must be of type VortexSetings")

        # instantiate BRAMs
        self.neurons = BRAM.Neurons(
            length=settings.amount_neurons,
            randomize_fun=self.rand_neuron
        )
        self.synapses = BRAM.Synapses(
            length=settings.amount_neurons,
            depth=settings.amount_synapses,
            word_size=np.uint8,
            randomize_fun=self.rand_synapse
        )

        self.active_neuron = settings.neuron
        
    def rand_neuron(self):
        return DataStructures.Neuron(
            param_leak_str=Utils.DiscreteNORM(2**3, 2, 0, 2**5),
            param_threshold=Utils.DiscreteNORM(2**6, 2**3, 0, 2**11),
            state_core=0,
            param_reset=Utils.DiscreteNORM(4., 1., 0, 2**3)
        )

    def rand_synapse(self, depth: int):
        temp_synapse = DataStructures.Synapse(
            length=depth,
            word_size=np.uint8
        )
        for i in range(depth):
            temp_synapse[i] = Utils.DiscreteNORM(0.5, 1., 0, 3)
        return temp_synapse
    

    