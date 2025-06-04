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
        sg_grad = 1.0 / (1.0 + torch.abs(input - threshold)) ** 2
        grad_input = grad_output * sg_grad
        grad_threshold = -grad_output * sg_grad
        return grad_input, grad_threshold

class SpikePooling2D(nn.Module):
    def __init__(
            self,
            num_channels: int,
            kernel_size: int = 2,
            stride: int = 2,
            decay: float = 0.05,
            threshold: float = 1.5,
            reset_value: float = 0.0
    ):
        super().__init__()
        self.decay = nn.Parameter(torch.full((num_channels,), decay))
        self.threshold = nn.Parameter(torch.full((num_channels,), threshold))
        self.kernel_size = kernel_size
        self.reset_value = reset_value
        self.stride = stride

    def forward(self, membrane, conv_out):
        membrane = membrane + conv_out
        decay = self.decay.view(1, -1, 1, 1)
        membrane = membrane - decay

        # Pool for spike output only
        pooled = F.avg_pool2d(membrane, kernel_size=self.kernel_size, stride=self.stride) * (self.kernel_size ** 2)
        spikes = SurrogateSpike.apply(pooled, self.threshold.view(1, -1, 1, 1))

        neurons_threshold_crossed = (membrane > self.threshold.view(1, -1, 1, 1))
        membrane = torch.where(neurons_threshold_crossed, torch.full_like(membrane, self.reset_value), membrane)

        return membrane, spikes  # keep membrane at original size!

class DebileClassifier(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.fc = nn.Linear(channels, num_classes, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=(2, 3))  # Average over spatial dimensions
        return self.fc(x)

class NetUtils():
    @staticmethod
    def beta_clamp(mem, beta):
        mem = torch.where(
            mem > 0,
            torch.clamp(mem - torch.abs(beta), min=0.0),
            mem
        )
        mem = torch.where(
            mem < 0,
            torch.clamp(mem + torch.abs(beta), max=0.0),
            mem
        )
        return mem

    @staticmethod
    def mem_clamp(mem, scale, multiplier, bits=12):
        max_val = (2**(bits - 1)) - 1
        max_val = max_val * scale / multiplier
        min_val = -(2**(bits - 1)) - 1
        min_val = min_val * scale / multiplier
        mem = torch.clamp(mem, min=min_val, max=max_val)
        return mem