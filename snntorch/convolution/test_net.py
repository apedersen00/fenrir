import torch
from f_quant_net import FQuantNetInt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = 20
x = torch.randn(1, T, 1, 32, 32, device=device)

model = FQuantNetInt().to(device)

with torch.no_grad():
    output = model(x)

print(f"Output shape: {output.shape}")  # Should be [1, num_classes]
print(f"Output: {output}")  # Print the output tensor