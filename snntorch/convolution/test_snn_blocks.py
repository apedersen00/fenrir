import torch, sys, io, contextlib
from snn_blocks import FeatureMapNeuronLayer, SumPooling2D, SurrogateSpike
from quant_blocks import QuantizedFeatureMap, QuantizedSumPooling2D

import warnings
warnings.filterwarnings("ignore", message="Named tensors and all their associated APIs are an experimental feature")


def test_surrogate_spike() -> bool:
    print("Testing SurrogateSpike...")
    x = torch.tensor([[0.8, 1.0, 1.2], [0.9, 1.1, 0.7]], requires_grad=True)
    threshold = torch.tensor(1.0)

    out = SurrogateSpike.apply(x, threshold)
    print("Spike output:", out)

    expected = (x >= threshold).float()
    if not torch.allclose(out.detach(), expected, atol=1e-3):
        print("SurrogateSpike output mismatch")
        return False

    loss = out.sum()
    loss.backward()
    print("Gradients:", x.grad)

    if torch.any(x.grad.abs() < 1e-7):
        print("Gradient vanishes")
        return False

    print("SurrogateSpike test passed.")
    return True

def test_feature_map_neuron_layer() -> bool:
    print("Testing FeatureMapNeuronLayer...")
    batch = 2
    channels = 3
    dimensions = (5,5)

    neuron = FeatureMapNeuronLayer(
        numFeatureMaps=channels,
        spatialShape=dimensions,
    )

    membrane = torch.zeros(batch, channels, *dimensions)
    input = torch.ones(batch, channels, *dimensions)
    update_membrane, spike = neuron(membrane, input)
    print("Update Membrane :" , update_membrane)
    print("Spike :" , spike)
    if update_membrane.shape != (batch, channels, *dimensions):
        print("Membrane shape mismatch")
        return False
    if spike.shape != (batch, channels, *dimensions):
        print("Spike shape mismatch")
        return False
    print("FeatureMapNeuronLayer test passed.")
    return True

def test_sum_pooling_2d() -> bool:
    print("Testing SumPooling2D...")
    batch = 2
    channels = 3
    H, W = 5, 5
    kernelSize = 2
    stride = 2

    pool = SumPooling2D(
        kernelSize=kernelSize,
        stride=stride,
        numFeatureMaps=channels,
    )

    x = torch.ones(batch, channels, H, W)
    spikes = pool(x)
    print("Spikes after pooling:", spikes)
    if spikes.shape != (batch, channels, 2, 2):
        print("Pooling output shape mismatch")
        return False
    print("SumPooling2D test passed.")
    return True

def test_quantized_feature_map() -> bool:
    print("Testing QuantizedFeatureMap...")

    batch = 2
    channels = 3
    dimensions = (5, 5)
    bit_width = 8
    qmin = 0
    qmax = (1 << bit_width) - 1

    fm = QuantizedFeatureMap(
        num_feature_maps=channels,
        spatial_shape=dimensions,
        bit_width=bit_width,
        init_threshold=100,   # Choose arbitrary, valid values
        init_decay=3,
        init_reset=0,
    )
    # Typical SNN: initial membrane is zero, input is random within bitwidth
    membrane = torch.zeros(batch, channels, *dimensions)
    input = torch.randint(low=50, high=150, size=(batch, channels, *dimensions))

    out_mem, out_spike = fm(membrane, input)
    print("Updated membrane:", out_mem)
    print("Spikes:", out_spike)

    # Check shape
    if out_mem.shape != (batch, channels, *dimensions):
        print("Membrane shape mismatch")
        return False
    if out_spike.shape != (batch, channels, *dimensions):
        print("Spike shape mismatch")
        return False
    # Check quantization range
    if not (out_mem.ge(qmin).all() and out_mem.le(qmax).all()):
        print("Membrane out of quantized range")
        return False
    # Spikes should be 0 or 1
    if not set(out_spike.flatten().tolist()).issubset({0, 1}):
        print("Spikes not binary")
        return False

    print("QuantizedFeatureMap test passed.")
    return True

def test_quantized_sum_pooling_2d() -> bool:
    print("Testing QuantizedSumPooling2D...")
    batch = 2
    channels = 3
    H, W = 6, 6
    kernel_size = 2
    stride = 2
    bit_width = 8

    pool = QuantizedSumPooling2D(
        kernel_size=kernel_size,
        stride=stride,
        num_feature_maps=channels,
        bit_width=bit_width,
        init_threshold=128,
    )

    x = torch.randint(50, 100, (batch, channels, H, W)).float()
    spikes = pool(x)
    print("Spikes:", spikes)

    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1
    if spikes.shape != (batch, channels, H_out, W_out):
        print("Spike shape mismatch")
        return False
    if not set(spikes.flatten().tolist()).issubset({0, 1}):
        print("Spikes not binary")
        return False

    print("QuantizedSumPooling2D test passed.")
    return True


def test_annotate(test: callable) -> str:

    if test():
        return f"✅ {test.__name__}" 
    else:
        return f"❌ {test.__name__}"


if __name__ == "__main__":
    verbose = "-v" in sys.argv

    if verbose:
        print("Running tests...")
    else:
        print("Tip: Run with -v flag for verbose output.")

    tests = [
        ("test_surrogate_spike", test_surrogate_spike),
        ("test_feature_map_neuron_layer", test_feature_map_neuron_layer),
        ("test_sum_pooling_2d", test_sum_pooling_2d),
        ("test_quantized_feature_map", test_quantized_feature_map),
        ("test_quantized_sum_pooling_2d", test_quantized_sum_pooling_2d),
    ]
    result = []
    for name, test in tests:
        if verbose:
            result.append(test_annotate(test))
        else:
            # Suppress print statements in tests by redirecting stdout
            with io.StringIO() as buf, contextlib.redirect_stdout(buf):
                result.append(test_annotate(test))
    for res in result:
        print(res)
