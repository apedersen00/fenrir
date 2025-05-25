import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.nn import QuantIdentity
from snn_blocks import SurrogateSpike

class QuantizedFeatureMap(nn.Module):
    def __init__(
            self,
            num_feature_maps:   int,
            spatial_shape:      tuple[int, int],
            bit_width:          int = 8,
            init_threshold:     float = 128.,
            init_decay:         float = 3.,
            init_reset:         float = 0.
    ):
        """
        args:
            num_feature_maps (int): Number of feature maps.
            spatial_shape (tuple[int, int]): Spatial dimensions of the feature maps.
            bit_width (int): Bit width for quantization.
            init_threshold (int): Initial threshold for quantization.
            init_decay (int): Initial decay for quantization.
            init_reset (int): Initial reset for quantization.
        """
        super().__init__()
        self.num_feature_maps = num_feature_maps
        self.spatial_shape = spatial_shape
        self.bit_width = bit_width
        
        self.qmin = 0
        self.qmax = (1 << bit_width) - 1
    
        self.threshold = nn.Parameter(torch.full((num_feature_maps,), init_threshold, dtype=torch.float32))
        self.decay = nn.Parameter(torch.full((num_feature_maps,), init_decay, dtype=torch.float32))
        self.reset = nn.Parameter(torch.full((num_feature_maps,), init_reset, dtype=torch.float32))

        # quantization for params
        self.quant_threshold = QuantIdentity(act_bit_width=bit_width)
        self.quant_decay = QuantIdentity(act_bit_width=bit_width)
        self.quant_reset = QuantIdentity(act_bit_width=bit_width)

        # quantization for the membrane
        self.quant_membrane = QuantIdentity(act_bit_width=bit_width)
    
    def forward(
            self,
            membrane: torch.Tensor,
            input: torch.Tensor
    ):
        decay = self.quant_decay(self.decay).view(1, -1, 1, 1)
        threshold = self.quant_threshold(self.threshold).view(1, -1, 1, 1)
        reset = self.quant_reset(self.reset).view(1, -1, 1, 1)

        membrane = membrane - decay + input
        membrane = self.quant_membrane(membrane)
        membrane = torch.clamp(membrane, self.qmin, self.qmax)

        spikes = SurrogateSpike.apply(membrane, threshold)

        membrane = torch.where(membrane >= threshold, reset, membrane)
        membrane = self.quant_membrane(membrane)
        membrane = torch.clamp(membrane, self.qmin, self.qmax)

        return membrane, spikes
    
class QuantizedSumPooling2D(nn.Module):
    def __init__(
            self,
            kernel_size:        int,
            stride:             int,
            num_feature_maps:   int,
            bit_width:          int = 8,
            init_threshold:     float = 128.
    ):
        """
        args:
            kernel_size (int): Size of the pooling kernel.
            stride (int): Stride for the pooling operation.
            num_feature_maps (int): Number of feature maps.
            bit_width (int): Bit width for quantization.
            init_threshold (float): Initial threshold for quantization.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_feature_maps = num_feature_maps
        self.bit_width = bit_width

        self.qmin = 0
        self.qmax = (1 << bit_width) - 1

        self.threshold = nn.Parameter(torch.full((num_feature_maps,), init_threshold, dtype=torch.float32))
        self.quant_threshold = QuantIdentity(act_bit_width=bit_width)

        self.quant_input = QuantIdentity(act_bit_width=bit_width)
        self.quant_output = QuantIdentity(act_bit_width=bit_width)

    def forward(self, x: torch.Tensor):

        x = self.quant_input(x) # x : (batch, channels, height, width)
        
        B, C, H, W = x.shape

        x_unfold = torch.nn.functional.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )

        K = self.kernel_size ** 2
        num_windows = x_unfold.shape[-1]
        x_unfold = x_unfold.view(B, C, K, num_windows)

        # Sum pooling
        pooled_sums = x_unfold.sum(dim=2)  # (B, C, num_windows)

        #Quantize pooled sums
        pooled_sums = self.quant_output(pooled_sums)
        pooled_sums = torch.clamp(pooled_sums, self.qmin, self.qmax)

        threshold = self.quant_threshold(self.threshold).view(1, -1, 1)
        spikes = SurrogateSpike.apply(pooled_sums, threshold)

        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        spikes2d = spikes.view(B, C, H_out, W_out)
        return spikes2d