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



def fake_quantize(
        x: torch.Tensor,
        quantization_bits: int = 8,
        signed: bool = False,
    ):
    """
    Fake quantization function that simulates the effect of quantization
    Args:
        x (torch.Tensor): Input tensor to be quantized.
        quantization_bits (int): Number of bits for quantization.
        signed (bool): Whether the quantization is signed or unsigned. (default: False)
    Returns:
        torch.Tensor: Quantized tensor.
    """

    if signed:
        qmin = -2 ** (quantization_bits - 1)
        qmax = 2 ** (quantization_bits - 1) - 1
    else:
        qmin = 0
        qmax = 2 ** quantization_bits - 1
    
    x = torch.clamp(x, qmin, qmax)
    x_q = torch.round(x)
    return x + (x_q - x).detach()  # Straight-through estimator (STE) for backpropagation

class QuantFeatureMap(nn.Module):
    def __init__(
            self,
            num_feature_maps: int,
            learnable: bool = True,
            signed: bool = False,
            bit_width: int = 8,
            init_threshold: float = 100.,
            init_decay: float = 1.,
            init_reset: float = 0.
    ):
        super().__init__()
        self.bitwidth = bit_width
        self.signed = signed

        self.threshold = nn.Parameter(
            torch.full((num_feature_maps,), float(init_threshold))
        ) if learnable else float(init_threshold)
        self.decay = nn.Parameter(
            torch.full((num_feature_maps,), float(init_decay))
        ) if learnable else float(init_decay)
        self.reset = nn.Parameter(
            torch.full((num_feature_maps,), float(init_reset))
        ) if learnable else float(init_reset)

    def forward(self, membrane: torch.tensor) -> torch.tensor:
        decay_q = fake_quantize(self.decay, self.bitwidth, self.signed).view(1, -1, 1, 1)
        threshold_q = fake_quantize(self.threshold, self.bitwidth, self.signed).view(1, -1, 1, 1)
        reset_q = fake_quantize(self.reset, self.bitwidth, self.signed).view(1, -1, 1, 1)

        membrane = membrane - decay_q
        membrane_q = fake_quantize(membrane, self.bitwidth, self.signed)

        spikes = SurrogateSpike.apply(membrane_q, threshold_q)

        membrane_q = torch.where(spikes.bool(), reset_q, membrane_q)
        membrane_q = fake_quantize(membrane_q, self.bitwidth, self.signed)

        return membrane_q

class QuantSumPooling(nn.Module):
    def __init__(
            self,
            num_feature_maps: int,
            kernel_size: int = 2,
            stride: int = 2,
            bitwidth: int = 8,
            signed: bool = False,
            learnable: bool = True,
            init_threshold: float = 100.,
            clamp_before_threshold: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.bitwidth = bitwidth
        self.signed = signed
        self.clamp_before_threshold = clamp_before_threshold

        self.qmin = -2 ** (bitwidth - 1) if signed else 0
        self.qmax = 2 ** (bitwidth - 1) - 1 if signed else 2 ** bitwidth - 1

        if learnable:
            self.threshold = nn.Parameter(torch.full((num_feature_maps,), init_threshold, dtype=torch.float32))
        else:
            self.register_buffer('threshold', torch.full((num_feature_maps,), init_threshold, dtype=torch.float32))

        
    def fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.round(x)
        return torch.clamp(x, self.qmin, self.qmax)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fake_quant(x)

        B, C, H, W = x.shape
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        K2 = self.kernel_size ** 2
        L = x_unfold.shape[-1]
        x_unfold = x_unfold.view(B, C, K2, L)
        pooled = x_unfold.sum(dim=2)

        if  self.clamp_before_threshold:
            pooled = self.fake_quant(pooled)
        
        threshold = self.threshold.view(1, -1, 1)
        spikes = SurrogateSpike.apply(pooled, threshold)
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1

        spikes2d = spikes.view(B, C, H_out, W_out)
        return spikes2d

class QuantConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: int,
            bias: bool,
            bit_width: int,
            signed: bool,
            quant_reg_strength: float
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            padding_mode='zeros'
        )
        self.bitwidth = bit_width
        self.signed = signed
        nn.init.constant_(self.conv.weight, 20.0)
        self.quant_reg_strength = quant_reg_strength
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        weight_q = fake_quantize(self.conv.weight, self.bitwidth, self.signed)
        self.quant_reg_loss = torch.mean((self.conv.weight - weight_q) ** 2)
        return F.conv2d(
            x,
            weight_q,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding
        )

class FQuantNetInt(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            conv1_out: int = 12,
            conv2_out: int = 24,
            kernel_size: int = 3,
            pool_kernel: int = 2,
            pool_stride: int = 2,
            num_classes: int = 10,
            bitwidth: int = 8,
            signed: bool = False,
            quant_reg_strength: float = 0.01,
    ):
        super().__init__()
        self.conv1 = QuantConv2d(
            in_channels, conv1_out, kernel_size=kernel_size,
            stride=1, padding=1, bias=False,
            bit_width=bitwidth, signed=signed,
            quant_reg_strength=quant_reg_strength
        )
        self.fmap1 = QuantFeatureMap(conv1_out, bit_width=bitwidth, signed=signed)
        self.pool1 = QuantSumPooling(conv1_out, kernel_size=pool_kernel, stride=pool_stride, bitwidth=bitwidth, signed=signed)

        # Conv2
        self.conv2 = QuantConv2d(
            conv1_out, conv2_out, kernel_size=kernel_size,
            stride=1, padding=1, bias=False,
            bit_width=bitwidth, signed=signed,
            quant_reg_strength=quant_reg_strength
        )
        self.fmap2 = QuantFeatureMap(conv2_out, bit_width=bitwidth, signed=signed)
        self.pool2 = QuantSumPooling(conv2_out, kernel_size=pool_kernel, stride=pool_stride, bitwidth=bitwidth, signed=signed)

        # Classifier
        self.fc = nn.Linear(conv2_out * 8 * 8, num_classes, bias=False)
        self.classifier_threshold = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        device = x.device
        mem1 = torch.zeros(B, self.conv1.conv.out_channels, H, W, device=device)
        mem2 = torch.zeros(B, self.conv2.conv.out_channels, H//2, W//2, device=device)

        logits = 0
        for t in range(T):
            frame = x[:, t, :, :, :]

            conv1_out = self.conv1(frame)
            mem1 = mem1 + conv1_out
            pooled1 = self.pool1(mem1)
            mem1 = self.fmap1(mem1)

            conv2_out = self.conv2(pooled1)
            mem2 = mem2 + conv2_out
            pooled2 = self.pool2(mem2)
            mem2 = self.fmap2(mem2)

            flat = pooled2.view(B, -1)
            fc_out = self.fc(flat)
            
            logits += fc_out

        return logits / T
    

class SpikePooling2D(nn.Module):
    def __init__(
            self,
            num_channels: int,
            kernel_size: int = 2,
            stride: int = 2,
            decay: float = 5.0,
            threshold: float = 100.0,
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

        return membrane, spikes  # keep membrane at original size!

class DebileClassifier(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.fc = nn.Linear(channels, num_classes, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=(2, 3))  # Average over spatial dimensions
        return self.fc(x)

class ThreeConvPoolingNet(nn.Module):
    def __init__(
            self,
            conv1_out: int = 12,
            conv2_out: int = 24,
            conv3_out: int = 8,
            kernel_size: int = 3
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv1_out, kernel_size=kernel_size, stride=1, padding=1, padding_mode='zeros', bias=False)
        self.conv2 = nn.Conv2d(in_channels=conv1_out, out_channels=conv2_out, kernel_size=kernel_size, stride=1, padding=1, padding_mode='zeros', bias=False)
        self.conv3 = nn.Conv2d(in_channels=conv2_out, out_channels=conv3_out, kernel_size=kernel_size, stride=1, padding=1, padding_mode='zeros', bias=False)
        
        nn.init.normal_(self.conv1.weight, mean=20.0, std=2.5)
        nn.init.normal_(self.conv2.weight, mean=20.0, std=2.5)
        nn.init.normal_(self.conv3.weight, mean=20.0, std=2.5)

        self.pool1 = SpikePooling2D(num_channels=conv1_out, kernel_size=2, stride=2)
        self.pool2 = SpikePooling2D(num_channels=conv2_out, kernel_size=2, stride=2)
        self.pool3 = SpikePooling2D(num_channels=conv3_out, kernel_size=2, stride=2)

        self.classifier = DebileClassifier(channels=conv3_out, num_classes=10)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, T, C, H, W = x.shape

        mem1 = torch.zeros(B, self.conv1.out_channels, H, W, device=x.device)
        mem2 = torch.zeros(B, self.conv2.out_channels, H//2, W//2, device=x.device)
        mem3 = torch.zeros(B, self.conv3.out_channels, H//4, W//4, device=x.device)

        logits = 0

        for t in range(T):

            xt = x[:, t, :, :, :]

            conv1_out = self.conv1(xt)
            mem1, spikes1 = self.pool1(mem1, conv1_out)

            conv2_out = self.conv2(spikes1)
            mem2, spikes2 = self.pool2(mem2, conv2_out)

            conv3_out = self.conv3(spikes2)
            mem3, spikes3 = self.pool3(mem3, conv3_out)

            out = self.classifier(spikes3)
            logits += out

        return logits / T
