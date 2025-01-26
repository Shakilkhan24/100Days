import torch
from torch.utils.cpp_extension import load

sources = ["binding.cpp", "ReLU.cu", "SoftMax.cu", "LeakyReLU.cu", "TanH.cu"]
functions = load("functions", sources=sources, verbose=True)

class CustomReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return functions.ReLU(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return functions.ReLUBackward(input, grad_output)

x = torch.tensor([-1.0, -1.0, -1.0, -2.0], device='cuda', requires_grad=True)

relu = CustomReLU.apply

y_custom = relu(x)
y_custom.sum().backward()  
grad_custom = x.grad.clone()  

x.grad.zero_()
y_pytorch = torch.nn.functional.relu(x)
y_pytorch.sum().backward()  
grad_pytorch = x.grad.clone()  

# Compare the gradients
print("Custom ReLU Gradient:", grad_custom)
print("PyTorch ReLU Gradient:", grad_pytorch)

if torch.allclose(grad_custom, grad_pytorch, atol=1e-6):
    print("Gradients match!")
else:
    print("Gradients do not match!")
