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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 4)  # 4 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# === 2. DATASET CLASS (Crop uniquement) ===
class CroppedImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.samples = [(path, label) for (path, label) in self.samples if "_crop" in path or "crop" not in root]
        self.imgs = self.samples  # compatibilité

# === 3. TRAINING FUNCTION ===
def train_model(dataset_path, save_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = CroppedImageFolder(dataset_path, transform=transform)
    print(f"\n📂 Dataset: {dataset_path}")
    print("✅ Classes détectées :", dataset.class_to_idx)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FruitMaturityCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
        print(f"📈 {os.path.basename(dataset_path)} | Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Modèle sauvegardé : {save_path}")

# === 4. MAIN ===
if __name__ == "__main__":
    configs = [
        ("dbsets/original", "model/cnn_original.pth"),
        ("dbsets/crop", "model/cnn_crop.pth"),
        ("dbsets/mixed", "model/cnn_mixed.pth"),
    ]

    for dataset_path, save_path in configs:
        train_model(dataset_path, save_path)

    print("\n🎉 Tous les modèles ont été entraînés et sauvegardés.")
