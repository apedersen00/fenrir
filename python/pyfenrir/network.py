import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
import snntorch as snn
import pyfenrir as fnr

class FenrirNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        num_classes     = int(config["num_classes"])
        w               = int(config["width"])
        h               = int(config["height"])
        conv1_out       = int(config["conv1_out"])
        conv2_out       = int(config["conv2_out"])
        conv3_out       = int(config["conv3_out"])
        kernel_size     = int(config["kernel_size"])
        fc1_bits        = int(config["fc1_bits"])
        fc2_bits        = int(config["fc2_bits"])
        fc1_beta_init   = config["fc1_beta"]
        fc2_beta_init   = config["fc2_beta"]
        fc1_thr_init    = config["fc1_thr"]
        fc2_thr_init    = config["fc2_thr"]
        self.fc1_multiplier  = config["fc1_multiplier"]

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv1_out, kernel_size=kernel_size, stride=1, padding=1, padding_mode='zeros', bias=False)
        self.conv2 = nn.Conv2d(in_channels=conv1_out, out_channels=conv2_out, kernel_size=kernel_size, stride=1, padding=1, padding_mode='zeros', bias=False)
        self.conv3 = nn.Conv2d(in_channels=conv2_out, out_channels=conv3_out, kernel_size=kernel_size, stride=1, padding=1, padding_mode='zeros', bias=False)
        
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)

        self.pool1 = fnr.SpikePooling2D(num_channels=conv1_out, kernel_size=2, stride=2)
        self.pool2 = fnr.SpikePooling2D(num_channels=conv2_out, kernel_size=2, stride=2)
        self.pool3 = fnr.SpikePooling2D(num_channels=conv3_out, kernel_size=2, stride=2)

        self.fc1_beta   = torch.nn.Parameter(torch.tensor(fc1_beta_init), requires_grad=True)
        self.fc1        = qnn.QuantLinear((h // 8) * (w // 8) * conv3_out, num_classes, bias=False, weight_bit_width=fc1_bits)
        self.lif1       = snn.Leaky(beta=1.0, threshold=fc1_thr_init, learn_threshold=True, reset_mechanism='zero', reset_delay=False)

    def forward(self, x: torch.Tensor):

        B, T, C, H, W = x.shape

        conv_mem1 = torch.zeros(B, self.conv1.out_channels, H, W, device=x.device)
        conv_mem2 = torch.zeros(B, self.conv2.out_channels, H//2, W//2, device=x.device)
        conv_mem3 = torch.zeros(B, self.conv3.out_channels, H//4, W//4, device=x.device)
        fc_mem1   = self.lif1.init_leaky()

        # Record output spikes
        spk_rec = []

        scale_fc1 = self.fc1.quant_weight().scale

        for t in range(T):

            xt = x[:, t, :, :, :]

            conv1_out = self.conv1(xt)
            conv_mem1, spk1 = self.pool1(conv_mem1, conv1_out)

            conv2_out = self.conv2(spk1)
            conv_mem2, spk2 = self.pool2(conv_mem2, conv2_out)

            conv3_out = self.conv3(spk2)
            conv_mem3, spk3 = self.pool3(conv_mem3, conv3_out)

            cur1 = self.fc1(spk3.view(B, -1))
            fc_mem1 = fnr.NetUtils.mem_clamp(fc_mem1, scale_fc1, multiplier=self.fc1_multiplier)
            spk4, fc_mem1 = self.lif1(cur1, fc_mem1)
            fc_mem1 = fnr.NetUtils.beta_clamp(fc_mem1, self.fc1_beta)

            spk_rec.append(spk4)

        return torch.stack(spk_rec)
