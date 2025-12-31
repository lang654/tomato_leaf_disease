# 🍅 Tomato Leaf Disease Detection (Few-Shot + TENT)

## 📌 Project Overview
This project proposes a robust tomato leaf disease detection framework that adapts from clean laboratory images to real-world field images using minimal labeled data.

The system combines:
- Vision Transformer (ViT)
- Few-Shot Fine-Tuning (10 images per class)
- Test-Time Adaptation (TENT)

This approach reduces the dependency on large labeled field datasets and avoids retraining at deployment.

---

## 📂 Datasets Used
- **PlantVillage** (Lab images)
- **PlantDoc** (Field images)

> Only 10 labeled field images per class were used for adaptation.

---

## 🧠 Methodology
1. Train ViT on clean lab images
2. Perform few-shot fine-tuning on field images
3. Apply Test-Time Entropy Minimization (TENT) using unlabeled field data
4. Evaluate performance on field test set

---

## 📊 Results
| Stage | Accuracy |
|------|----------|
| Lab Training | 91.85% |
| Few-Shot Field | 13.30% |
| Few-Shot + TENT | **15.60%** |

---

## 🛠 Tech Stack
- Python
- PyTorch
- Vision Transformer (ViT)
- timm
- scikit-learn
- matplotlib

---

## 🚀 How to Run
```bash
python train_vit_lab.py
python few_shot_finetune.py
python tent_adaptation.py
python final_evaluation.py
```

## 📦 Full dataset available here (academic use only):  
https://drive.google.com/drive/folders/1L4_btGLgu6rpKcVIYCA5zTHYkZjHvHOa?usp=drive_link
