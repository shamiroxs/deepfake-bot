import torch
from torchsummary import summary

# Load your model (this is just an example, replace it with your own model loading logic)
model_path = "./model/efficientnet_v2_s.pth"

# Load the pre-trained EfficientNetV2 model and update the classifier
original_model = torch.hub.load('hankyul2/EfficientNetV2-pytorch', 'efficientnet_v2_s', pretrained=False)
original_model.load_state_dict(torch.load(model_path))

# Save the full architecture of the model to 'model.txt'
with open("model.txt", "w") as file:
    # Save the model's architecture as a string
    print(str(original_model), file=file)

print("Model architecture saved to 'model.txt'")

