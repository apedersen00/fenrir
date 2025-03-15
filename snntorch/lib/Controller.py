import numpy as np
from hwcomponents import *
from dataclasses import dataclass
from typing import Callable, List, Optional, Union
import copy, os, datetime

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
        self.input_data = settings.InputData

        self.log = []
        self.spike_log = []

    def rand_neuron(self):
        return DataStructures.Neuron(
            param_leak_str=Utils.DiscreteNORM(2**3, 2, 0, 2**5),
            param_threshold=Utils.DiscreteNORM(2**4, 2**2, 0, 2**11),
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
    
    def simulate(self):
        for sample in self.input_data:
            self.sim_sample(sample)
            


    def sim_sample(self, sample: DataStructures.InputSample):
        counter_neuron, counter_synapse = 0, 0

        temp_log = []
        

        for neuron in self.neurons:
            
            spike_event = 0

            self.active_neuron.change_neuron(neuron)

            for synapse in self.synapses[counter_neuron]:
                self.active_neuron.change_weight(synapse)
                x = self.active_neuron.forward(sample[counter_synapse])
                if x==1:
                    spike_event = 1
                
                counter_synapse += 1
                
            counter_neuron += 1
            counter_synapse = 0

            temp_log.append([self.active_neuron.neuron.core, spike_event])
        
        self.log.append(temp_log)
        

class Export(VortexOne):
    def __init__(
            self, 
            settings: VortexSetings,
            word_sizes: dict[str, int],
            dir_name: Optional[str] = None
            ):
        super().__init__(settings)
        self.original_neurons = copy.deepcopy(self.neurons)
        self.word_sizes = word_sizes
        self.dir_name = dir_name
        
        #self.export()

    def export(self):
        
        if self.make_dir():
            print("Directory created")
        else:
            return False
        
        if self.export_neurons(self.original_neurons, "start_neurons"):
            print("Neurons exported")
        else:
            return False
        
        if self.export_synapses():
            print("Synapses exported")
        else:
            return False

        self.simulate()

        if self.export_neurons(self.neurons, "end_neurons"):
            print("Neurons exported")
        else:
            return False
        
        # need to export input sample and ouputs from the simulation


    def export_input(self) -> bool:

        print("Exporting input data")
        print(self.input_data)

        pass

    def make_dir(self) -> bool:
        if self.dir_name is None:
            # Create a directory named export_current_date_hour_minute
            self.dir_name = f"export_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
        
        if os.path.exists(self.dir_name):
            print(f"Directory {self.dir_name} already exists")
            return False
        else:
            try:
                os.mkdir(self.dir_name)
                os.chdir(self.dir_name)
                return True
            except:
                print(f"Error creating directory {self.dir_name}")
                return False
        
    def export_neurons(self, neurons: BRAM.Neurons, filename: str) -> bool:

        words: List[str] = []
        wz = self.word_sizes
        
        conv = self.convert_int_to_bin
        for neuron in neurons:
            leak_str = conv(int(neuron.leak_str), wz["param_leak_str"])
            threshold = conv(int(neuron.threshold), wz["param_threshold"])
            reset = conv(int(neuron.reset), wz["param_reset"])
            state_core = conv(int(neuron.core), wz["state_core"])

            #make sure the final word is the correct size
            word = f"{state_core}{reset}{threshold}{leak_str}"
            word = "0" * (wz["bram_neuron"] - len(word)) + word
            words.append(word)
            
        return self.export_list_to_file(
            data = words,
            filename=f"{filename}.mem"
        )

    def export_synapses(self) -> bool:
        
        #create a ref to the synapses bram object
        synapses = self.synapses
        synapse_width = self.word_sizes["synapses_width"]
        bram_width = self.word_sizes["bram_synapse"]

        weights_per_address = bram_width // synapse_width
        print(f"weights per address: {weights_per_address}")
        
        addresses_per_neuron = synapses.length / weights_per_address
        if addresses_per_neuron % 1 != 0:
            addresses_per_neuron = int(addresses_per_neuron) + 1
        else:
            addresses_per_neuron = int(addresses_per_neuron)

        print(f"addresses per neuron: {addresses_per_neuron}")

        words: str = []

        for synapse in synapses:
            tmp_word = ""
            for i in range(synapse.length):
                tmp_word = self.convert_int_to_bin(synapse[i], synapse_width) + tmp_word
                if (i + 1) % weights_per_address == 0:
                    words.append(tmp_word)
                    tmp_word = ""
                # if we are at the last synapse, we need to add the last word
                if i == synapse.length - 1:
                    # also add zeros to fill the last word
                    tmp_word = "0" * (bram_width - len(tmp_word)) + tmp_word
                    words.append(tmp_word)

        return self.export_list_to_file(
            data = words,
            filename="synapses.mem"
        )
        
        pass

    def convert_int_to_bin(self, value: int, width: int) -> str:
        return bin(value)[2:].zfill(width)
    
    def export_list_to_file(self, data: List[str], filename: str) -> bool:
        try:
            with open(filename, "w") as f:
                f.write("\n".join(data))  # No extra newline at the end
            return True
        except Exception as e:
            print(f"Error writing to file {filename}: {e}")
            return False