import torch
import torch.nn.functional as F

# Set the seed for reproducibility
torch.manual_seed(42)

# Define the dimensions
seq_len = 4
dim = 4

# Initialize the tensors
Q = torch.ones(seq_len, dim, requires_grad=True)
K = torch.ones(seq_len, dim, requires_grad=True)
V = torch.ones(seq_len, dim, requires_grad=True)

# Forward pass
scores = torch.matmul(Q, K.transpose(-2, -1)) / (dim ** 0.5)
P = F.softmax(scores, dim=-1)
O = torch.matmul(P, V)

# Create a dummy gradient for the output
dO = torch.ones_like(O)

# Backward pass
O.backward(dO)


print("PyTorch O:")
print(O)
# Print the gradients
print("PyTorch dQ:")
print(Q.grad)
print("PyTorch dK:")
print(K.grad)
print("PyTorch dV:")
print(V.grad)