import torch
import torch.nn as nn

# When a dataset is imbalanced, 
# you can give more weight to errors on underrepresented classes.
class_weights = torch.tensor(
    [1.0, 1.0, 1.0, 1.0, 1.0,
     1.0, 1.0, 1.0, 1.0, 1.0],

# These equal weights are just a demonstration and do not change the relative contribution of the CIFAR-10 classes. 
# For an imbalanced dataset, calculate appropriate weights from the training-set class frequencies only.

    dtype=torch.float32,
    device=torch.device
)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

# inspect misclassified images
# A useful next step is to record images for which the predicted label differs from the actual label.
@torch.no_grad()

def find_misclassified(model, dataloader, device):
    model.eval()
    errors = []

    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        wrong = predictions != labels.to(device)

        for i in wrong.nonzero(as_tuple=True)[0]:
            errors.append({
                "image": images[i].cpu(),
                "actual": labels[i].item(),
                "predicted": predictions[i].item()
            })

    return errors

# The returned images are still normalized tensors. To display them correctly, reverse the normalization before plotting.
# Error analysis can reveal confusing classes, insufficient training data, or preprocessing mistakes.
