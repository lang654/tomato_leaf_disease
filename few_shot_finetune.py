import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os

# ================= CONFIG =================
BASE_DIR = r"C:\Users\subha\Downloads\Final Year Project\project datasets"
FEWSHOT_DIR = os.path.join(BASE_DIR, "field", "few_shot_train")

MODEL_PATH = "vit_lab_source.pth"
SAVE_PATH = "vit_fewshot.pth"

BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
# =========================================

print("Using device:", DEVICE)

# ================= TRANSFORMS =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# ============================================

# ================= DATA =================
dataset = datasets.ImageFolder(FEWSHOT_DIR, transform=transform)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

num_classes = len(dataset.classes)
print("Few-shot classes:", dataset.classes)
# =======================================

# ================= MODEL =================
model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=False,
    num_classes=num_classes  # FIELD classes = 8
)

# Load LAB weights EXCEPT classifier head
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

# Remove classifier weights
state_dict.pop("head.weight", None)
state_dict.pop("head.bias", None)

model.load_state_dict(state_dict, strict=False)
model = model.to(DEVICE)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze classifier head
for param in model.head.parameters():
    param.requires_grad = True

# Unfreeze last transformer block
for param in model.blocks[-1].parameters():
    param.requires_grad = True


# ================= TRAINING =================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    acc = 100. * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(loader):.4f} Acc: {acc:.2f}%")

# ================= SAVE =================
torch.save(model.state_dict(), SAVE_PATH)
print(f"💾 Few-shot model saved as {SAVE_PATH}")
