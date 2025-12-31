import os
import random
import shutil

# ================== CONFIG ==================
BASE_DIR = r"C:\Users\subha\Downloads\Final Year Project\project datasets"
SOURCE_DIR = os.path.join(BASE_DIR, "field", "train")
FEWSHOT_DIR = os.path.join(BASE_DIR, "field", "few_shot_train")
TEST_DIR = os.path.join(BASE_DIR, "field", "test")

K = 10  # few-shot samples per class
random.seed(42)
# ============================================

print("SOURCE DIR:", SOURCE_DIR)

# Safety check
if not os.path.exists(SOURCE_DIR):
    raise FileNotFoundError(f"❌ Source directory not found: {SOURCE_DIR}")

os.makedirs(FEWSHOT_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Loop through each disease class
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"\nProcessing class: {class_name}")
    print(f"Total images found: {len(images)}")

    if len(images) == 0:
        print("⚠️ Skipping empty class")
        continue

    # Select few-shot images
    if len(images) <= K:
        selected = images
        print("⚠️ Less than 10 images — using all")
    else:
        selected = random.sample(images, K)

    # Create destination folders
    os.makedirs(os.path.join(FEWSHOT_DIR, class_name), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, class_name), exist_ok=True)

    # Move images
    for img in images:
        src = os.path.join(class_path, img)
        if img in selected:
            dst = os.path.join(FEWSHOT_DIR, class_name, img)
        else:
            dst = os.path.join(TEST_DIR, class_name, img)
        shutil.move(src, dst)

    print(f"✅ Few-shot: {len(selected)} | Test: {len(images) - len(selected)}")

print("\n🎉 Few-shot split completed successfully!")
