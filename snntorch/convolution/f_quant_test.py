import tonic.datasets as datasets
import tonic.transforms as transforms
from tonic import DiskCachedDataset
import torch
from torch.utils.data import DataLoader
import tonic
from f_quant_net import QuantFeatureMap

class CropTo32(object):
    def __call__(self, frames):
        return frames[..., 1:33, 1:33]

class OnlyPositive(object):
    def __call__(self, frames):
        return frames[:, 1:2, :, :]



# Use native sensor size!
frame_transform = transforms.Compose([
    transforms.ToFrame(sensor_size=tonic.datasets.NMNIST.sensor_size, n_time_bins=200),
    CropTo32(),
    OnlyPositive(),
])

trainset = datasets.NMNIST(
    save_to='./data/nmnist',
    train=True,
    transform=frame_transform,
)


train_loader = DataLoader(
    trainset,
    batch_size=32,
    shuffle=True,
    collate_fn=tonic.collation.PadTensors()
)

for frames, targets in train_loader:
    print(f"Frames shape: {frames.shape}")    # [B, T, 1, 32, 32]
    print(f"Targets shape: {targets.shape}")  # [B]
    break

batch = next(iter(train_loader))
frames, targets = batch
print(frames.shape, frames.dtype, frames.min(), frames.max())
print(targets)

def test_fake_quant_feature_map():
    torch.manual_seed(0)
    B, C, H, W = 2, 4, 3, 3  # Small test shapes
    bitwidth = 4  # Test with tiny bitwidth for easier debugging

    fmap = QuantFeatureMap(
        num_feature_maps=C,
        bit_width=bitwidth,
        learnable=True,
        init_decay=4.0,
        init_threshold=9.5,
        init_reset=0.0,
        signed=False
    )

    # Initialize membrane and input to known values
    membrane = torch.full((B, C, H, W), 9.5)  # Over threshold
    input_tensor = torch.ones(B, C, H, W) * 2

    # Forward
    mem_out = fmap(membrane, input_tensor)

    print("Output membrane after fake quantization:\n", mem_out)
    print("Membrane min/max:", mem_out.min().item(), mem_out.max().item())

    # Check that everything is within quantized range
    qmin, qmax = 0, 2**bitwidth - 1
    assert torch.all(mem_out >= qmin) and torch.all(mem_out <= qmax), "Quantization/clamping failed"
    print("Test passed.")

test_fake_quant_feature_map()
