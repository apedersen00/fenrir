import torch
from snn_blocks import FeatureMapNeuronLayer, SumPooling2D
import sys
import io
import contextlib

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
        ("test_feature_map_neuron_layer", test_feature_map_neuron_layer),
        ("test_sum_pooling_2d", test_sum_pooling_2d),
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
