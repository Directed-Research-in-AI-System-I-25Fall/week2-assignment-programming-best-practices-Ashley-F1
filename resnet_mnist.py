import torch
from torchvision import datasets, transforms
from transformers import ResNetForImageClassification
from torch.utils.data import DataLoader

# Load the pretrained ResNet model
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
model.eval()

# Resize MNIST images to 224x224, RGB channels to fit the ResNet model
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load MNIST dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Inference
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        
        # Get predictions
        outputs = model(images).logits
        _, predicted = torch.max(outputs, 1)
        
        # Calculate accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")