import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os

# ================== CONFIG ==================
BASE_DIR = r"C:\Users\subha\Downloads\Final Year Project\project datasets"
LAB_TRAIN = os.path.join(BASE_DIR, "lab", "train")
LAB_TEST  = os.path.join(BASE_DIR, "lab", "test")

BATCH_SIZE = 16
EPOCHS = 15
LR = 3e-4
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_SAVE_PATH = "vit_lab_source.pth"
# ============================================

print("Using device:", DEVICE)

# ================== TRANSFORMS ==================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# ===============================================

# ================== DATA ==================
train_dataset = datasets.ImageFolder(LAB_TRAIN, transform=train_transform)
test_dataset  = datasets.ImageFolder(LAB_TEST, transform=test_transform)

num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)
# =========================================

# ================== MODEL ==================
model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=num_classes
)
model = model.to(DEVICE)
# ===========================================

# ================== LOSS & OPTIMIZER ==================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)
# =====================================================

# ================== TRAIN LOOP ==================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    train_acc = 100. * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {running_loss/len(train_loader):.4f} "
          f"Train Acc: {train_acc:.2f}%")

# ================== EVALUATION ==================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

test_acc = 100. * correct / total
print(f"\n✅ LAB Test Accuracy: {test_acc:.2f}%")

# ================== SAVE MODEL ==================
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"💾 Model saved as {MODEL_SAVE_PATH}")
