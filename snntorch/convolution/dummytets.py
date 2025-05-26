import torch
import torch.nn as nn

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

class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold = nn.Parameter(torch.ones(3))
    def forward(self, x):
        thr = self.threshold.view(1, -1, 1, 1)
        return SurrogateSpike.apply(x, thr)

model = DummyNet()
x = torch.randn(2, 3, 4, 4)
out = model(x)
loss = out.sum()
loss.backward()
print("Grad:", model.threshold.grad)
