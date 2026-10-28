# load libraries
import torch
from resnet_model import build_resnet18, freeze_backbone

# load the model architecture
model = build_resnet18(num_classes=10, pretrained=True)
# freeze the pre-trained model
model = freeze_backbone(model)

x = torch.randn(2, 3, 224, 224)
y = model(x)

print("Input Shape: ", x.shape)
print("Output Shape: ", y.shape)

trainable = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print("Trainable Parameters: ", trainable)

# The input has two images, three color channels, and dimensions of 224 × 224. 
# The output contains ten class scores for each image.