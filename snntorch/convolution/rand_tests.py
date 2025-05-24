import torch.nn as nn
import torch
from snn_blocks import FeatureMapNeuronLayer, SumPooling2D, SurrogateSpike
from conv_nets import NMNISTV2
# convtest = nn.Conv2d(1, 1, 3, padding=1)

# # input tensor of shape [batch_size, channels, height, width]
# input_tensor = torch.randn(1, 1, 32, 32)

# # output tensor
# output_tensor = convtest(input_tensor)
# print("Input shape:", input_tensor.shape)
# print("Output shape:", output_tensor.shape)
# pool = SumPooling2D(kernelSize=2, stride=2, numFeatureMaps=1)
# # input tensor of shape [batch_size, channels, height, width]
# input_tensor = torch.randn(1, 1, 32, 32)
# # output tensor
# output_tensor = pool(input_tensor)
# print("Input shape:", input_tensor.shape)
# print("Output shape:", output_tensor.shape)

#test simple first with the layer used in the network to debug
# create input tensor
input_tensor = torch.randn(1, 1, 32, 32)
print("Created input_tensor with shape:", input_tensor.shape)

# create conv2d layer
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=1)
print("Initialized Conv2d layer:", conv2d)

# apply conv2d layer
output_tensor = conv2d(input_tensor)
print("After Conv2d, output_tensor shape:", output_tensor.shape)

# apply on feature map neuron layer
fmap = FeatureMapNeuronLayer(numFeatureMaps=1, spatialShape=(32, 32))
print("Initialized FeatureMapNeuronLayer:", fmap)

membrane = torch.zeros(1, 1, 32, 32)
print("Initialized membrane with zeros, shape:", membrane.shape)

output_tensor, spikes = fmap(membrane, output_tensor)
print("After FeatureMapNeuronLayer, output_tensor shape:", output_tensor.shape)

# apply on sum pooling layer
pool = SumPooling2D(kernelSize=2, stride=2, numFeatureMaps=1)
print("Initialized SumPooling2D layer:", pool)
output_tensor = pool(output_tensor)
print("After SumPooling2D, output_tensor shape:", output_tensor.shape)

# apply second conv2d layer
conv2d_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=1)
print("Initialized second Conv2d layer:", conv2d_2)
output_tensor = conv2d_2(output_tensor)
print("After second Conv2d, output_tensor shape:", output_tensor.shape)
# apply on feature map neuron layer
fmap2 = FeatureMapNeuronLayer(numFeatureMaps=1, spatialShape=(16, 16))
print("Initialized second FeatureMapNeuronLayer:", fmap2)
membrane = torch.zeros(1, 1, 16, 16)
print("Initialized membrane with zeros, shape:", membrane.shape)
output_tensor, spikes = fmap2(membrane, output_tensor)
print("After second FeatureMapNeuronLayer, output_tensor shape:", output_tensor.shape)
# apply on sum pooling layer
pool2 = SumPooling2D(kernelSize=2, stride=2, numFeatureMaps=1)
print("Initialized second SumPooling2D layer:", pool2)
output_tensor = pool2(output_tensor)
print("After second SumPooling2D, output_tensor shape:", output_tensor.shape)
# apply on linear layer
linear = nn.Linear(1 * 8 * 8, 10)
print("Initialized Linear layer:", linear)
output_tensor = output_tensor.view(1, -1)  # Flatten the tensor
print("Flattened output_tensor shape:", output_tensor.shape)
output_tensor = linear(output_tensor)
print("After Linear layer, output_tensor shape:", output_tensor.shape)

# make a new test with the same, but for more channels
# create input tensor
input_tensor = torch.randn(1, 3, 32, 32)
print("Created input_tensor with shape:", input_tensor.shape)
# create conv2d layer
conv2d = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, stride=1)
print("Initialized Conv2d layer:", conv2d)
# apply conv2d layer
output_tensor = conv2d(input_tensor)
print("After Conv2d, output_tensor shape:", output_tensor.shape)
# apply on feature map neuron layer
fmap = FeatureMapNeuronLayer(numFeatureMaps=3, spatialShape=(32, 32))
print("Initialized FeatureMapNeuronLayer:", fmap)
membrane = torch.zeros(1, 3, 32, 32)
print("Initialized membrane with zeros, shape:", membrane.shape)
output_tensor, spikes = fmap(membrane, output_tensor)
print("After FeatureMapNeuronLayer, output_tensor shape:", output_tensor.shape)
# apply on sum pooling layer
pool = SumPooling2D(kernelSize=2, stride=2, numFeatureMaps=3)
print("Initialized SumPooling2D layer:", pool)
output_tensor = pool(output_tensor)
print("After SumPooling2D, output_tensor shape:", output_tensor.shape)

# apply second conv2d layer
conv2d_2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1, stride=1)
print("Initialized second Conv2d layer:", conv2d_2)
output_tensor = conv2d_2(output_tensor)
print("After second Conv2d, output_tensor shape:", output_tensor.shape)
# apply on feature map neuron layer
fmap2 = FeatureMapNeuronLayer(numFeatureMaps=6, spatialShape=(16, 16))
print("Initialized second FeatureMapNeuronLayer:", fmap2)
membrane = torch.zeros(1, 6, 16, 16)
print("Initialized membrane with zeros, shape:", membrane.shape)
output_tensor, spikes = fmap2(membrane, output_tensor)
print("After second FeatureMapNeuronLayer, output_tensor shape:", output_tensor.shape)
# apply on sum pooling layer
pool2 = SumPooling2D(kernelSize=2, stride=2, numFeatureMaps=6)
print("Initialized second SumPooling2D layer:", pool2)
output_tensor = pool2(output_tensor)
print("After second SumPooling2D, output_tensor shape:", output_tensor.shape)
# apply on linear layer
linear = nn.Linear(6 * 8 * 8, 10)
print("Initialized Linear layer:", linear)
output_tensor = output_tensor.view(1, -1)  # Flatten the tensor
print("Flattened output_tensor shape:", output_tensor.shape)
output_tensor = linear(output_tensor)
print("After Linear layer, output_tensor shape:", output_tensor.shape)
