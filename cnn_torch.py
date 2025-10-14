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
    transforms.ToTensor(),
])

# ---------- 3. Load dataset ----------
train_dir = "data/train"
test_dir = "data/test"

train_data = datasets.ImageFolder(root=train_dir, transform=transform)
test_data = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

# ---------- 4. Define CNN ----------
class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CatDogCNN().to(device)
print(model)

# ---------- 5. Loss & optimizer ----------
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

# ---------- 6. Training loop ----------
epochs = 5
print("\n--- Training ---")
torch.cuda.synchronize() if device.type == "cuda" else None
train_start = time.time()

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

torch.cuda.synchronize() if device.type == "cuda" else None
train_end = time.time()
train_time = train_end - train_start
print(f"Training time: {train_time:.2f} seconds\n")

# ---------- 7. Evaluation ----------
print("--- Evaluation ---")
torch.cuda.synchronize() if device.type == "cuda" else None
eval_start = time.time()

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

torch.cuda.synchronize() if device.type == "cuda" else None
eval_end = time.time()
eval_time = eval_end - eval_start
accuracy = correct / total

print(f"Test Accuracy: {accuracy*100:.2f}%")
print(f"Evaluation time: {eval_time:.2f} seconds")
print("\n--- Summary ---")
print(f"Device: {device}")
print(f"Training time: {train_time:.2f} sec | Evaluation time: {eval_time:.2f} sec | Accuracy: {accuracy*100:.2f}%")