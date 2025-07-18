import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image


# === 1. CNN MODEL ===
class FruitMaturityCNN(nn.Module):
    def __init__(self):
        super(FruitMaturityCNN, self).__init__()
        # First convolutional layer: input channels = 3 (RGB), output = 32 feature maps
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # Second convolutional layer: input = 32, output = 64 feature maps
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Max pooling layer with 2x2 kernel
        self.pool = nn.MaxPool2d(2, 2)
        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.25)
        # Fully connected layer (after flattening conv output)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        # Output layer for 4 classes
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        # Apply convolution -> ReLU -> pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the feature maps
        x = x.view(-1, 64 * 32 * 32)
        # Apply dropout and fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# === 2. DATASET CLASS (Crop only if in path) ===
class CroppedImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        # Keep only cropped images (i.e., filenames containing '_crop')
        self.samples = [
            (path, label)
            for (path, label) in self.samples
            if "_crop" in path or "crop" not in root
        ]
        self.imgs = self.samples  # for compatibility with older versions


# === 3. TRAINING FUNCTION ===
def train_model(dataset_path, save_path):
    # Image preprocessing pipeline
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),  # Resize all images to 128x128
            transforms.ToTensor(),  # Convert to PyTorch tensor
            transforms.Normalize([0.5], [0.5]),  # Normalize pixel values
        ]
    )

    # Load dataset from the given path
    dataset = CroppedImageFolder(dataset_path, transform=transform)
    print(f"\n📂 Dataset: {dataset_path}")
    print("✅ Detected classes:", dataset.class_to_idx)

    # Split dataset into 80% train / 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FruitMaturityCNN().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop for 10 epochs
    for epoch in range(10):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            f"📈 {os.path.basename(dataset_path)} | Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}"
        )

    # Save the trained model
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved at: {save_path}")


# === 4. MAIN ===
if __name__ == "__main__":
    # Define paths for training on three datasets: original, crop, mixed
    configs = [
        ("dbsets/original", "model/cnn_original.pth"),
        ("dbsets/crop", "model/cnn_crop.pth"),
        ("dbsets/mixed", "model/cnn_mixed.pth"),
    ]

    # Train and save model for each dataset
    for dataset_path, save_path in configs:
        train_model(dataset_path, save_path)

    print("\n🎉 All models have been trained and saved.")
