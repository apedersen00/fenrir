from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from matplotlib import pyplot as plt
from typing import Callable, List, Optional


class DataStructures:
    
    @dataclass
    class Neuron:
        def __init__(
                self,
                param_leak_str:     int,
                param_threshold:    int,
                param_reset:        int,
                state_core:         int
        ):
            self.leak_str   = param_leak_str
            self.threshold  = param_threshold
            self.reset      = param_reset
            self.core       = state_core
        def __repr__(self):
            return f"Neuron For Memory(leak_str={self.leak_str}, threshold={self.threshold}, reset={self.reset}, core={self.core})"

    @dataclass
    class Synapse:
        length: int
        word_size: np.dtype

        def __post_init__(self):
            self.weights = np.zeros(self.length, dtype=self.word_size)

        def __getitem__(self, key):
            return self.weights[key]
        
        def __setitem__(self, key, value):
            self.weights[key] = value

        def __len__(self):
            return self.length

        def __repr__(self):
            return f"Synapse For Memory(length={self.length}, weights={self.weights})"
        
    @dataclass
    class InputSample:
        data: np.ndarray

        def __init__(
                self,
                length: int,
                data: np.ndarray = None,
                randomize_fun: Callable[[int], np.ndarray] = None
        ):
            if data is not None:
                self.data = data
            elif randomize_fun:
                self.data = randomize_fun(length)
            else:
                raise ValueError("Either 'data' or 'randomize_fun' must be provided")

        def __getitem__(self, key):
            return self.data[key]
        
        def __setitem__(self, key, value):
            self.data[key] = value

        def __len__(self):
            return len(self.data)

        def __repr__(self):
            return f"InputSample(input_data={self.data})"
        
    @dataclass
    class InputData:
        samples: List[DataStructures.InputSample] = field(default_factory=list)
        
        def add_sample(self, sample: DataStructures.InputSample):
            self.samples.append(sample)

        def __getitem__(self, index: int) -> DataStructures.InputSample:
            return self.samples[index]

        def __len__(self):
            return len(self.samples)

        def __iter__(self):
            return iter(self.samples)

        def __repr__(self):
            return f"InputSampleCollection(samples={self.samples})"


class Utils:

    @staticmethod
    def DiscreteNORM(
        mu: float,
        sigma: float,
        min: int,
        max: int
    ) -> int:
        """
        Returns a discrete sample from a normal distribution
        Clips the output to the range [min, max]
        """
        sample = np.random.normal(loc=mu, scale=sigma)
        return np.clip(
            np.round(sample),
            min, max
        )
    
    class Print:
        @staticmethod
        def PrettyNeuron(
            neuron: DataStructures.Neuron
        ) -> None:
            print(
                f"Neuron state and params:",
                f"\n - Leak strength: \t{neuron.leak_str}",
                f"\n - threshold: \t\t{neuron.threshold}",
                f"\n - reset value: \t{neuron.reset}",
                f"\n - core state: \t\t{neuron.core}",
            )
    
    class Plotting:

        @staticmethod
        def input_data_as_heatmap(input_data: DataStructures.InputData):
            """Plots the InputData as a heatmap."""
            data_matrix = np.array([sample.data for sample in input_data.samples])  # Convert to 2D array
            
            plt.figure(figsize=(10, 6))
            plt.imshow(data_matrix, cmap="coolwarm", aspect="auto", interpolation="nearest")  # Heatmap

            # Labels and formatting
            plt.colorbar(label="Value Intensity")
            plt.xlabel("Input Index")
            plt.ylabel("Sample Index")
            plt.title("InputData Heatmap")
            plt.xticks(np.arange(data_matrix.shape[1]))
            plt.yticks(np.arange(data_matrix.shape[0]))

            plt.show()
    
class BRAM:

    @dataclass
    class Neurons:
        def __init__(
            self,
            length: int,
            randomize_fun: Callable[[], DataStructures.Neuron] = None
        ):
            self.length = length
            self.neurons = [DataStructures.Neuron(0, 0, 0, 0) for _ in range(length)]

            if randomize_fun:
                self.__randomize(randomize_fun)

        def __randomize(self, callback):
            for i in range(self.length):
                self.neurons[i] = callback()
            
        def __getitem__(self, key):
            return self.neurons[key]
        
        def __setitem__(self, key, value):
            self.neurons[key] = value

        def __len__(self):
            return len(self.neurons)
        
        def __iter__(self):
            return iter(self.neurons)

        def __repr__(self):
            return f"BRAM_Neuron(neurons={self.neurons})"
        
    @dataclass
    class Synapses:
        def __init__(
            self,
            length: int,
            depth: int,
            word_size: np.dtype,
            randomize_fun: Callable[[int], DataStructures.Synapse] = None
        ):
            self.length = length
            self.depth = depth
            self.synapses = [DataStructures.Synapse(length, word_size) for _ in range(length)]

            if randomize_fun:
                self.__randomize(randomize_fun)
            
        def __randomize(self, callback):
            for i in range(self.length):
                self.synapses[i] = callback(self.depth)
        
        def __getitem__(self, address, key: Optional[int] = None):

            if key:
                return self.synapses[address][key]
            else:
                return self.synapses[address]
            
        
        def __setitem__(self, key, value):
            self.synapses[key] = value

        def __len__(self):
            return (self.length, self.depth)
        
        def __repr__(self):
            return f"BRAM_synaptic_weights(synapses={self.synapses})"
        
class Neurons:

    class LIF:

        neuron: DataStructures.Neuron
        weight: int

        def change_neuron(self, neuron: DataStructures.Neuron):
            self.neuron = neuron

        def change_weight(self, weight: int):
            self.weight = weight
        
        def forward(self, input):
            spike_event = 0
            current_state = self.neuron.core
            current_state += input * self.weight

            if current_state >= self.neuron.threshold:
                current_state = self.neuron.reset
                spike_event = 1

            self.neuron.core = current_state

            return spike_event