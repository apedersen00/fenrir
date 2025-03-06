from Neurons import LIFNeuron
import torch

num_synapses = 5
neuron = LIFNeuron(num_synapses)

# Loss function (Binary Cross Entropy for spike outputs)
optimizer = torch.optim.Adam(neuron.parameters(), lr=0.01)

# Training Data (Random binary input and desired spike output)
inputs = torch.randint(0, 2, (10, num_synapses), dtype=torch.float32)  # 10 samples
targets = torch.randint(0, 2, (10, 1), dtype=torch.float32)  # 10 desired spike outputs

# Training loop
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()  # Reset gradients
    
    total_loss = 0
    for i in range(len(inputs)):
        loss = neuron.compute_surrogate_loss(inputs[i], targets[i])  # Compute loss using surrogate function
        loss.backward()  # Backpropagation
        total_loss += loss.item()

    optimizer.step()  # Update weights
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(inputs)}")

print("Training Complete!")
print("input", inputs)
print("output", targets)

#print a prediction for all inputs and the target
for i in range(len(inputs)):
    print(f"Prediction: {neuron(inputs[i]).item()}, Target: {targets[i].item()}")