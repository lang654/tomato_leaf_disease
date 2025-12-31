import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import numpy as np
import matplotlib.pyplot as plt

# ================= CONFIG =================
BASE_DIR = r"C:\Users\subha\Downloads\Final Year Project\project datasets"
FIELD_TEST = os.path.join(BASE_DIR, "field", "test")

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
class_names = dataset.classes
# =======================================

def evaluate(model_path, title):
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=len(class_names)
    )

    state_dict = torch.load(model_path, map_location=DEVICE)
    state_dict.pop("head.weight", None)
    state_dict.pop("head.bias", None)

    model.load_state_dict(state_dict, strict=False)
    model = model.to(DEVICE)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    acc = accuracy_score(y_true, y_pred)
    print(f"{title} Accuracy: {acc*100:.2f}%")

    cm = confusion_matrix(y_true, y_pred)
    return cm, acc


# ================= EVALUATION =================
cm_fewshot, acc_fewshot = evaluate("vit_fewshot.pth", "Few-shot")
cm_tent, acc_tent = evaluate("vit_fewshot.pth", "After TENT")
# ==============================================

# ================= CONFUSION MATRIX =================
fig, ax = plt.subplots()
ax.imshow(cm_tent)
ax.set_title("Confusion Matrix (After TENT)")
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
plt.tight_layout()
plt.show()
