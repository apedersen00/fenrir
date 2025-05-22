import torch
import torch.nn as nn
import torch.nn.functional as F

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input, threshold)
        return (input >= threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, threshold = ctx.saved_tensors
        # Fast sigmoid surrogate: d/dx σ(x-thr) ~ 1/(1+|x-thr|)^2
        sg_grad = 1.0 / (1.0 + torch.abs(input - threshold)) ** 2
        return grad_output * sg_grad, None  # one grad per input
    
class FeatureMapNeuronLayer(nn.Module):

    def __init__(
            self,
            numFeatureMaps: int,
            spatialShape: tuple,
            learnable: bool = True,
            initThreshold: float = 1.0,
            initDecay: float = 0.05,
            initReset: float = 0.0
    ):
        super().__init__()
        self.numFeatureMaps = numFeatureMaps
        self.spatialShape = spatialShape
        
        # make learnable parameters
        self.threshold = nn.Parameter(
            torch.full((numFeatureMaps,), initThreshold)
        ) if learnable else initThreshold
        self.decay = nn.Parameter(
            torch.full((numFeatureMaps,), initDecay)
        ) if learnable else initDecay
        self.reset = nn.Parameter(
            torch.full((numFeatureMaps,), initReset)
        ) if learnable else initReset

    def forward(self, membrane, input):
        # Membrane [B, C, H, W] 
        # Input [B, C, H, W]
        decay = self.decay.view(1, -1, 1, 1)
        threshold = self.threshold.view(1, -1, 1, 1)
        reset = self.reset.view(1, -1, 1, 1)

        membrane = membrane - decay + input
        spikes = SurrogateSpike.apply(membrane, threshold)

        membrane = torch.where(membrane >= threshold, reset, membrane)
        return membrane, spikes
    
class SumPooling2D(nn.Module):
    
    def __init__(
            self,
            kernelSize: int,
            stride: int,
            numFeatureMaps: int,
            initThreshold: float = 1.0,
            learnable: bool = True,
    ):
        super().__init__()
        self.kernelSize = kernelSize
        self.stride = stride

        # learnable parameters
        if learnable:
            self.threshold = nn.Parameter(
                torch.full((numFeatureMaps,), initThreshold)   
            )          
        else:
            #if not, use 
            self.register_buffer(
                'threshold', torch.full((numFeatureMaps,), initThreshold)
            )
        
    def forward(self, x):
        # x : [B, C, H, W]
        batchSize, numChannels, height, width = x.shape

        xUnfold = F.unfold(
            x,
            kernel_size=self.kernelSize,
            stride=self.stride,
        )

        windowArea = self.kernelSize ** 2
        numWindows = xUnfold.shape[-1]

        xUnfold = xUnfold.view(
            batchSize,
            numChannels,
            windowArea,
            numWindows
        )

        # Thresholding
        pooledSums = xUnfold.sum(dim=2)
        threshold = self.threshold.view(1, -1, 1)
        
        spikes = SurrogateSpike.apply(pooledSums, threshold)

        # reshape to 2d 
        heightOut = (height - self.kernelSize) // self.stride + 1
        widthOut = (width - self.kernelSize) // self.stride + 1

        spikes2D = spikes.view(
            batchSize,
            numChannels,
            heightOut,
            widthOut
        )
        return spikes2D

