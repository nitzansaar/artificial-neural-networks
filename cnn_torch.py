import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# ---------- 1. Device setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.backends.cudnn.benchmark = True

# ---------- 2. Data transforms ----------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),               # scales to [0,1]
])

# ---------- 3. Load dataset ----------
train_dir = "data/train"
test_dir = "data/test"

train_data = datasets.ImageFolder(root=train_dir, transform=transform)
test_data = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(
    train_data,
    batch_size=256,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
test_loader = DataLoader(
    test_data,
    batch_size=256,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

# ---------- 4. Define CNN ----------
class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 -> 16 -> 32 channels
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 1)      # binary logit output

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # logits
        return x

model = CatDogCNN().to(device)
print(model)

# ---------- 5. Loss & optimizer ----------
criterion = nn.BCEWithLogitsLoss()            # numerically stable
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

# ---------- 6. Training loop ----------
epochs = 5
start = time.time()
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f}")
print("Training time:", time.time() - start, "seconds")

# ---------- 7. Evaluation ----------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).squeeze().long()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
accuracy = correct / total
print(f"Test Accuracy: {accuracy*100:.2f}%")