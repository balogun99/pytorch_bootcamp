# Part 1 — Advanced CNN architectures
# import libraries
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# build the ResNet18 model
def build_resnet18(num_classes=10, pretrained=True):
    # load the pre-trained ResNet18 model
    if pretrained:
        weights = ResNet18_Weights.DEFAULT # loads pretrained weights when pretrained=True. The initial download requires an internet connection.
    else:
        weights = None

    model = resnet18(weights=weights) # loads the ResNet-18 architecture.

    # replace the original ImageNet classifier
    in_features = model.fc.in_features # obtains the number of features entering the original classifier.
    model.fc = nn.Linear(in_features, num_classes) # replaces the 1,000-class ImageNet classifier with a 10-class CIFAR-10 classifier.

    return model # The model returns raw logits, so we can use nn.CrossEntropyLoss() without adding Softmax to the model.

# freeeze the pretrained model
def freeze_backbone(model):

    for param in model.parameters():
        param.requires_grad = False

    # allow the new classifier to learn
    for param in model.fc.parameters():
            param.requires_grad = True

    return model