import torch
import onnx
from onnx2torch import convert

# Load ONNX model
onnx_model = onnx.load("./model/efficientnetv2.onnx")

# Convert ONNX to PyTorch model
torch_model = convert(onnx_model)

# Save as PyTorch .pth
torch.save(torch_model.state_dict(), "efficientnetv2.pth")

print("Converted ONNX to PyTorch .pth format")

