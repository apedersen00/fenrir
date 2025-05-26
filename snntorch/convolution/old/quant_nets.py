import torch
import torch.nn as nn
import brevitas.nn as qnn  # for QuantConv2d if you want quantized weights!
from quant_blocks import QuantizedFeatureMap, QuantizedSumPooling2D

class QuantizedNMNISTV2(nn.Module):
    def __init__(
        self,
        in_channels=1,
        conv1_out=12,
        conv2_out=24,
        conv_kernel=3,
        pool_kernel=2,
        pool_stride=2,
        num_classes=10,
        input_hw=34,  # Default: 34
        bit_width=8,
        fmap1_thresh=128.,
        fmap1_decay=3.,
        fmap1_reset=0.,
        fmap2_thresh=128.,
        fmap2_decay=3.,
        fmap2_reset=0.,
        pool1_thresh=128.,
        pool2_thresh=128.,
    ):
        super().__init__()

        # Use quantized Conv2d (with weight QAT) if you want:
        self.conv1 = qnn.QuantConv2d(
            in_channels, conv1_out, kernel_size=conv_kernel, padding=1,
            weight_bit_width=bit_width, bias=False
        )
        self.conv2 = qnn.QuantConv2d(
            conv1_out, conv2_out, kernel_size=conv_kernel, padding=1,
            weight_bit_width=bit_width, bias=False
        )

        self.fmap1 = QuantizedFeatureMap(
            num_feature_maps=conv1_out,
            spatial_shape=(input_hw, input_hw),
            bit_width=bit_width,
            init_threshold=fmap1_thresh,
            init_decay=fmap1_decay,
            init_reset=fmap1_reset,
        )
        self.pool1 = QuantizedSumPooling2D(
            kernel_size=pool_kernel,
            stride=pool_stride,
            num_feature_maps=conv1_out,
            bit_width=bit_width,
            init_threshold=pool1_thresh,
        )

        self.fmap2 = QuantizedFeatureMap(
            num_feature_maps=conv2_out,
            spatial_shape=(input_hw//2, input_hw//2),
            bit_width=bit_width,
            init_threshold=fmap2_thresh,
            init_decay=fmap2_decay,
            init_reset=fmap2_reset,
        )
        self.pool2 = QuantizedSumPooling2D(
            kernel_size=pool_kernel,
            stride=pool_stride,
            num_feature_maps=conv2_out,
            bit_width=bit_width,
            init_threshold=pool2_thresh,
        )

        # Classifier - you may want qnn.QuantLinear if you want QAT here too!
        self.fc = nn.Linear(conv2_out * 8 * 8, num_classes)

    def forward(self, x):
        batch_size, num_steps, _, H, W = x.shape
        device = x.device

        mem1 = torch.zeros(batch_size, self.conv1.out_channels, H, W, device=device)
        mem2 = torch.zeros(batch_size, self.conv2.out_channels, H//2, W//2, device=device)

        spk_outs = []

        for t in range(num_steps):
            frame = x[:, t, :, :, :]  # [B, 1, H, W]

            fmap1 = self.conv1(frame)
            pooled1 = self.pool1(fmap1)
            mem1, _ = self.fmap1(mem1, fmap1)  # Optionally: use spikes if needed

            fmap2 = self.conv2(pooled1)
            pooled2 = self.pool2(fmap2)
            mem2, _ = self.fmap2(mem2, fmap2)

            flat = pooled2.view(batch_size, -1)  # flatten for FC
            out = self.fc(flat)
            spk_outs.append(out)

        out = torch.stack(spk_outs, dim=0).sum(dim=0)
        return out
