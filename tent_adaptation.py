import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os

# ================= CONFIG =================
BASE_DIR = r"C:\Users\subha\Downloads\Final Year Project\project datasets"
FIELD_TEST = os.path.join(BASE_DIR, "field", "test")

MODEL_PATH = "vit_fewshot.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_WORKERS = 0
# =========================================

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# ============================================

# ================= DATA =================
dataset = datasets.ImageFolder(FIELD_TEST, transform=transform)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)
num_classes = len(dataset.classes)
print("Field test classes:", dataset.classes)
# =======================================

# ================= MODEL =================
model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=False,
    num_classes=num_classes  # 7 test classes
)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

# Remove classifier head (class count mismatch)
state_dict.pop("head.weight", None)
state_dict.pop("head.bias", None)

model.load_state_dict(state_dict, strict=False)
model = model.to(DEVICE)
model.train()
# ========================================

# ================= TENT (ViT-compatible) =================
# Freeze everything
for param in model.parameters():
    param.requires_grad = False

# Enable LayerNorm adaptation
ln_params = []
for m in model.modules():
    if isinstance(m, nn.LayerNorm):
        for p in m.parameters():
            p.requires_grad = True
            ln_params.append(p)

print(f"LayerNorm params to adapt: {len(ln_params)}")

optimizer = torch.optim.Adam(ln_params, lr=1e-3)

print("Running TENT adaptation (LayerNorm)...")

for images, _ in loader:
    images = images.to(DEVICE)

    outputs = model(images)

    entropy = -torch.sum(
        torch.softmax(outputs, dim=1) *
        torch.log_softmax(outputs, dim=1),
        dim=1
    ).mean()

    optimizer.zero_grad()
    entropy.backward()
    optimizer.step()

print("✅ TENT adaptation completed (ViT)")
# ========================================================

