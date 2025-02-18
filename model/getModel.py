import torch

# Load EfficientNetV2 model
model = torch.hub.load('hankyul2/EfficientNetV2-pytorch', 'efficientnet_v2_s', pretrained=True, nclass=1000)

# Save the model to the current directory
torch.save(model.state_dict(), "efficientnet_v2_s.pth")
print("Model saved in current directory as efficientnet_v2_s.pth")

