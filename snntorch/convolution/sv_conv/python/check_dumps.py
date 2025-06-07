import numpy as np
import pandas as pd
import os, torch

dump_folder = "./mem_dumps/"
dump_prefix = "memory_dump_event_"

def load_events_file():
    with open("test_events.txt", "r") as f:
        lines = f.readlines()[2:]
        events = [tuple(map(int, line.strip().split(','))) for line in lines if line.strip()]
    return events



if __name__ == "__main__":
    print("Checking if the dumps are correct...")
    events = load_events_file()
    print(f"Number of events: {len(events)}")
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    event_img = np.zeros((32, 32), dtype=np.float32)

    hw_kernel_ch0 = np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=np.float32)
    hw_kernel_ch1 = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]
        ], dtype=np.float32)
    hw_kernel = np.stack((hw_kernel_ch0, hw_kernel_ch1), axis=0)
    hw_kernel = hw_kernel[:, np.newaxis, :, :]    
    
    weight_tensor = torch.from_numpy(hw_kernel).float()
    conv = torch.nn.Conv2d(
            in_channels=1, out_channels=2,
            kernel_size=3, padding=1, bias=False, padding_mode='zeros'
        )
    conv.weight.data = weight_tensor

    for idx, event in enumerate(events):

        mem_dump = pd.read_csv(
            dump_folder + dump_prefix + str(idx+1) + ".csv",
            header=0
        )
        
        dump = mem_dump['ch0'].to_numpy()
        dump = dump.reshape((32,32,))
        dump2 = mem_dump['ch1'].to_numpy()
        dump2 = dump2.reshape((32,32,))
        
        #print(f"Dump {idx}:\n{dump}")
        event_img[event[0], event[1]] += 1.0
        img_tensor = torch.from_numpy(event_img).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            out_tensor = conv(img_tensor)
        out_img = out_tensor.squeeze().numpy()
        # cast torch floats to integers
        out_img = out_img.astype(np.int32)

        if np.array_equal(dump, out_img[0]) and np.array_equal(dump2, out_img[1]):
            print(f"Dump {idx} is correct.")

        if idx == 0:
            print(f"Dump {idx}:\n{dump}")
            print(f"Output {idx}:\n{out_img[1]}")