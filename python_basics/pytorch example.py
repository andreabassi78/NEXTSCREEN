'''
Convolutional Neural Network for digit recognition.
An image of the CNN is here:
https://raw.githubusercontent.com/andreabassi78/NEXTSCREEN/refs/heads/future/images/cnn.webp
For intro on NN with pytorch:
https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define a simple Convolutional Neural Network for number recognition
class NumberRecognitionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) # input channel 1 (grayscale), output channels 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # input channel 32, output channels 64
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x))) # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define data transformations: Preprocess the dataset to convert images to tensors and normalize pixel values
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL images or NumPy arrays to PyTorch tensors with values scaled between 0 and 1
    transforms.Normalize((0.5,), (0.5,))  # Normalizes pixel values to have mean 0.5 and standard deviation 0.5 (approximately scaling them between -1 and 1)
])

# Load MNIST dataset and create a subset of the training dataset
full_train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
subset_size = 5000  # Use only 5000 of the 60000 images downloaded from MNIST
train_dataset = Subset(full_train_dataset, list(range(subset_size)))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Function to show a few images from the dataset
def show_images(dataset, title):
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for i, ax in enumerate(axes):
        image, label = dataset[i]
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title(f'Label: {label}')
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

# Show images from the training and test datasets
show_images(train_dataset, "Training Dataset Examples")
show_images(test_dataset, "Test Dataset Examples")

# Instantiate the model, define the loss and optimizer
model = NumberRecognitionCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
correct = 0
for images, labels in test_loader:
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    correct += (predicted == labels).sum().item()

accuracy = 100 * correct / len(test_dataset)
print(f"Accuracy: {accuracy:.2f}%")