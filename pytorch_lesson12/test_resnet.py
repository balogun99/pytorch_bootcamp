# load libraries
import torch

from models.resnet_model import (
    build_resnet18,
    freeze_backbone
)

# configure available device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)

print(
    "Device:", device
)

# build the pre-trained model
model = build_resnet18(
    num_classes=10,
    pretrained=True
)

# freeze the pretrained backbone
model = freeze_backbone(model)
model = model.to(device)

# create a batch of two dummy RGB images
images = torch.randn(2, 3, 224, 224).to(device) # shape of the tensor images

# run a forward pass
model.eval()

with torch.no_grad():
    outputs = model(images)

print("Input Shape: ", images.shape)
print("Output Shape: ", outputs.shape)

# count trainable parameters
trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(
    "Trainable Parameters :", trainable_parameters
)

assert outputs.shape == (2, 10)
print (
    "ResNet18 Test PASSED.!"
)