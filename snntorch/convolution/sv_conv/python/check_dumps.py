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

    event_img = np.zeros((8, 8), dtype=np.float32)

    hw_kernel = np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=np.float32)
    weight_tensor = torch.from_numpy(hw_kernel).unsqueeze(0).unsqueeze(0)
    conv = torch.nn.Conv2d(
            in_channels=1, out_channels=1,
            kernel_size=3, padding=1, bias=False, padding_mode='zeros'
        )
    conv.weight.data = weight_tensor

    for idx, event in enumerate(events):

        mem_dump = pd.read_csv(
            dump_folder + dump_prefix + str(idx+1) + ".csv",
            header=0
        )
        
        dump = mem_dump['ch0'].to_numpy()
        dump = dump.reshape((8,8,))
        #print(f"Dump {idx}:\n{dump}")
        event_img[event[0], event[1]] += 1.0
        img_tensor = torch.from_numpy(event_img).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            out_tensor = conv(img_tensor)
        out_img = out_tensor.squeeze().numpy()
        # cast torch floats to integers
        out_img = out_img.astype(np.int32)

        if np.array_equal(dump, out_img):
            print(f"Dump {idx} is correct.")