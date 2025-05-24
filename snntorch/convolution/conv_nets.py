import torch
import torch.nn as nn
from snn_blocks import FeatureMapNeuronLayer, SumPooling2D
import snntorch as snn

class NMNISTProofNet(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 num_feature_maps=4, 
                 conv_kernel=3, 
                 pool_kernel=2, 
                 pool_stride=2, 
                 num_classes=10, 
                 input_hw=34):  # <- Set default to 34
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_feature_maps, kernel_size=conv_kernel, stride=1, padding=1)  # padding=1

        conv_hw = input_hw  # padding keeps it same
        pooled_hw = (conv_hw - pool_kernel) // pool_stride + 1
        self.pool = SumPooling2D(kernelSize=pool_kernel, stride=pool_stride, numFeatureMaps=num_feature_maps)
        self.fc = nn.Linear(num_feature_maps * pooled_hw * pooled_hw, num_classes)
        self.lif = snn.Leaky(beta=0.95)

    def forward(self, x):
        # x: [B, T, C, H, W]
        batch_size, num_steps, _, H, W = x.shape
        device = x.device

        mem = self.lif.init_leaky()  # initialize membrane state
        spk_rec = []

        for t in range(num_steps):
            frame = x[:, t, :, :, :]  # [B, 1, 32, 32]
            fmap = self.conv(frame)  # [B, num_feature_maps, H', W']
            pooled = self.pool(fmap)  # [B, num_feature_maps, pooled_H, pooled_W]
            flat = pooled.view(batch_size, -1)  # flatten for FC
            fc_out = self.fc(flat)
            spk, mem = self.lif(fc_out, mem)  # update membrane state
            spk_rec.append(spk)

        # Stack spikes over time: [T, B, num_classes] → [B, T, num_classes]
        spk_rec = torch.stack(spk_rec, dim=0).permute(1, 0, 2)
        # Sum spikes over time (could also use mem, etc.)
        out = spk_rec.sum(dim=1)
        return out  # [B, num_classes]

    def get_tensor_stats(tensor):
        
        return {
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'nonzero': torch.count_nonzero(tensor).item(),
            'total': tensor.numel(),
            'sparsity': 1.0 - torch.count_nonzero(tensor).item() / tensor.numel()
        }
    

class NMNISTV2(nn.Module):
    def __init__(
            self,
            in_channels = 1,
            conv1_out = 12,
            conv2_out = 24,
            conv_kernel = 3,
            pool_kernel = 2,
            pool_stride = 2,
            num_classes = 10,
            input_hw = 34,  # <- Set default to 34
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_out, kernel_size=conv_kernel, padding=1)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=conv_kernel, padding=1)

        self.fmap1 = FeatureMapNeuronLayer(numFeatureMaps=conv1_out, spatialShape=(input_hw, input_hw))
        self.fmap2 = FeatureMapNeuronLayer(numFeatureMaps=conv2_out, spatialShape=(input_hw, input_hw//2))

        self.pool1 = SumPooling2D(kernelSize=pool_kernel, stride=pool_stride, numFeatureMaps=conv1_out)
        self.pool2 = SumPooling2D(kernelSize=pool_kernel, stride=pool_stride, numFeatureMaps=conv2_out)

        #classifier
        self.fc = nn.Linear(24 * 8 * 8 , num_classes)
        
    def forward(self, x):

        batch_size, num_steps, _, H, W, = x.shape
        device = x.device

        mem1 = torch.zeros(batch_size, self.conv1.out_channels, H, W, device=device)
        mem2 = torch.zeros(batch_size, self.conv2.out_channels, H//2, W//2, device=device)

        spk_outs = []

        for t in range (num_steps):

            frame = x[:, t, :, :, :]  # [B, 1, 32, 32]

            fmap1 = self.conv1(frame)
            pooled1 = self.pool1(fmap1)
            mem1, spikes1 = self.fmap1(mem1, fmap1) # spikes from feature maps are not used
            
            fmap2 = self.conv2(pooled1)
            pooled2 = self.pool2(fmap2)
            mem2, spikes2 = self.fmap2(mem2, fmap2)
            
            flat = pooled2.view(batch_size, -1)  # flatten for FC
            out = self.fc(flat)
            spk_outs.append(out)

        out = torch.stack(spk_outs, dim=0).sum(dim=0)
        return out
    
    @staticmethod
    def get_tensor_stats(tensor):
        return {
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'nonzero': torch.count_nonzero(tensor).item(),
            'total': tensor.numel(),
            'sparsity': float(1.0 - torch.count_nonzero(tensor).item() / tensor.numel())
        }


class NMNISTFORPLOTTING(nn.Module):
    def __init__(
            self,
            in_channels = 1,
            conv1_out = 12,
            conv2_out = 24,
            conv_kernel = 3,
            pool_kernel = 2,
            pool_stride = 2,
            num_classes = 10,
            input_hw = 34,  # <- Set default to 34
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_out, kernel_size=conv_kernel, padding=1)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=conv_kernel, padding=1)

        self.fmap1 = FeatureMapNeuronLayer(numFeatureMaps=conv1_out, spatialShape=(input_hw, input_hw))
        self.fmap2 = FeatureMapNeuronLayer(numFeatureMaps=conv2_out, spatialShape=(input_hw, input_hw//2))

        self.pool1 = SumPooling2D(kernelSize=pool_kernel, stride=pool_stride, numFeatureMaps=conv1_out)
        self.pool2 = SumPooling2D(kernelSize=pool_kernel, stride=pool_stride, numFeatureMaps=conv2_out)

        #classifier
        self.fc = nn.Linear(24 * 8 * 8 , num_classes)
        
    def forward(self, x, return_features=False):

        batch_size, num_steps, _, H, W, = x.shape
        device = x.device

        mem1 = torch.zeros(batch_size, self.conv1.out_channels, H, W, device=device)
        mem2 = torch.zeros(batch_size, self.conv2.out_channels, H//2, W//2, device=device)

        spk_outs = []
        mem1_list, mem2_list = [], []

        for t in range (num_steps):

            frame = x[:, t, :, :, :]  # [B, 1, 32, 32]

            fmap1 = self.conv1(frame)
            pooled1 = self.pool1(fmap1)
            mem1, spikes1 = self.fmap1(mem1, fmap1) # spikes from feature maps are not used
            
            fmap2 = self.conv2(pooled1)
            pooled2 = self.pool2(fmap2)
            mem2, spikes2 = self.fmap2(mem2, fmap2)
            
            flat = pooled2.view(batch_size, -1)  # flatten for FC
            out = self.fc(flat)
            spk_outs.append(out)

            if return_features:
                mem1_list.append(mem1.detach().cpu().clone())
                mem2_list.append(mem2.detach().cpu().clone())

        out = torch.stack(spk_outs, dim=0).sum(dim=0)
        if return_features:
            return out, torch.stack(mem1_list), torch.stack(mem2_list)
        return out