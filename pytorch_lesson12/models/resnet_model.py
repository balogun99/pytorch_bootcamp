import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18(num_classes=10, pretrained=True):
    """
    Build a ResNet-18 model for CIFAR-10 classification.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        A ResNet-18 model with a new classification layer.
    """

    weights = ResNet18_Weights.DEFAULT if pretrained else None

    model = resnet18(weights=weights)

    # Replace the original 1000-class ImageNet classifier.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def freeze_backbone(model):
    """Freeze the backbone and train only the final classifier."""

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    return model


def set_feature_extraction_mode(model):
    """
    Keep frozen backbone BatchNorm layers in evaluation mode.
    The new classifier remains in training mode.
    """

    model.train()

    # Prevent frozen BatchNorm running statistics from changing.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    model.fc.train()

    return model

def fine_tune_last_block(model):

    """
    Freeze the entire model, then unfreeze layer4
    and the final classification layer.
    """

    for parameter in model.parameters():
        parameter.require_grad = False

    for parameter in model.layer4.parameters():
        parameter.require_grad = True

    for parameter in model.fc.parameters():
        parameter.require_grad = True

    return model

# What this code does:
# - build_resnet18() loads ResNet-18 and replaces its original 1,000-class classifier with a 10-class classifier.
# - freeze_backbone() freezes the pretrained network so only the new classifier's parameters are updated.
# - set_feature_extraction_mode() keeps frozen BatchNorm layers from updating their running statistics during training.


# # Part 1 — Advanced CNN architectures
# # import libraries
# import torch.nn as nn
# from torchvision.models import resnet18, ResNet18_Weights

# # ResNet Preprocessing
# weights = ResNet18_Weights.DEFAULT
# preprocess = weights.transforms()

# # build the ResNet18 model
# def build_resnet18(num_classes=10, pretrained=True):
#     # load the pre-trained ResNet18 model
#     if pretrained:
#         weights = ResNet18_Weights.DEFAULT # loads pretrained weights when pretrained=True. The initial download requires an internet connection.
#     else:
#         weights = None

#     model = resnet18(weights=weights) # loads the ResNet-18 architecture.

#     # replace the original ImageNet classifier
#     in_features = model.fc.in_features # obtains the number of features entering the original classifier.
#     model.fc = nn.Linear(in_features, num_classes) # replaces the 1,000-class ImageNet classifier with a 10-class CIFAR-10 classifier.

#     return model # The model returns raw logits, so we can use nn.CrossEntropyLoss() without adding Softmax to the model.

# # freeeze the pretrained model
# def freeze_backbone(model):

#     for param in model.parameters():
#         param.requires_grad = False

#     # allow the new classifier to learn
#     for param in model.fc.parameters():
#             param.requires_grad = True

#     return model