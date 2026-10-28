# We need to remember something important:
# CIFAR-10 images are:
# 32 × 32 RGB

# Our prediction pipeline therefore needs:
# Image
#  ↓
# Resize 32×32
#  ↓
# Tensor
#  ↓
# Normalize
#  ↓
# Add batch dimension
#  ↓
# CNN
#  ↓
# Prediction

# import libraries
import torch

from PIL import Image
from torchvision import transforms
from models.cnn import CIFAR10CNN

# list the dataset classes
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# configure hardware device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# step 8
# load cifar10 model
model = CIFAR10CNN(
    num_classes=10
)

model.load_state_dict(
    torch.load(
        "pytorch_lesson10/checkpoints/best_model.pth",
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)

model.eval()

# transform the image to tensor and normalize it
transform = transforms.Compose([
    
    transforms.Resize((32, 32)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.4914, 0.4822, 0.4465],
        [0.2470, 0.2435, 0.2616]
    )
])

# predict function
def predict_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(
        0
    )

    image_tensor = image_tensor.to(
        device
    )

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0, predicted_class
        ].item()

        print(
            f"Prediction: "
            f"This is a/an {classes[predicted_class]}"
        )

        print(
            f"Confidence: "
            f"{confidence * 100:.2f}%"
        )

if __name__ == "__main__":

    image_path = input(
        "Enter image path: "
    )

    predict_image(
        image_path
    )