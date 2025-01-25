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
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define data transformations: convert images to tensors and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset and create a subset of the training dataset
full_train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
subset_size = 20000
train_dataset = Subset(full_train_dataset, list(range(subset_size)))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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

# Visualize Feature Maps (Method 2)
image, _ = train_dataset[0]
image = image.unsqueeze(0)  # Add batch dimension

feature_map1 = model.conv1(image)  # Feature map after the first convolution
feature_map2 = model.conv2(model.pool(torch.relu(feature_map1)))  # After the second convolution

# Visualize feature maps from the second convolutional layer
feature_map = feature_map2.detach().squeeze().cpu().numpy()
fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i < feature_map.shape[0]:
        ax.imshow(feature_map[i], cmap='gray')
        ax.axis('off')
plt.suptitle("Feature Maps from the Second Convolutional Layer")
plt.show()

# Feature Visualization (Method 3)
input_image = torch.randn(1, 1, 28, 28, requires_grad=True)
optimizer = optim.Adam([input_image], lr=0.1)

for i in range(100):
    optimizer.zero_grad()
    activation_value = model.conv2(model.pool(torch.relu(model.conv1(input_image)))).mean()
    (-activation_value).backward()
    optimizer.step()

optimized_image = input_image.detach().squeeze().cpu().numpy()
plt.imshow(optimized_image, cmap='gray')
plt.title("Visualization of a Kernel from Conv2")
plt.axis('off')
plt.show()

# Visualize Kernels of the First Convolutional Layer
kernels = model.conv1.weight.data
kernels = (kernels - kernels.min()) / (kernels.max() - kernels.min())

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    if i < kernels.shape[0]:
        kernel = kernels[i, 0].cpu().numpy()
        ax.imshow(kernel, cmap='gray')
        ax.axis('off')
plt.suptitle("Kernels of the First Convolutional Layer")
plt.show()
