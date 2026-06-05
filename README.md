# Biased & Noisy Dataset Analysis for Object Recognition

A study of how **class imbalance** and **data noise** harm an image
classifier, and how to fix them. The task is binary image classification
(cats vs dogs); the focus is on diagnosing dataset problems and applying
targeted solutions: data cleaning, class weights, data augmentation and
**transfer learning with ResNet50**.

> Academic project (Advanced Machine Learning, Taif University).
> author: **Arwa Saad Alsawat** 
---

## The problem

Two data-quality issues were identified and addressed:

- **Class imbalance** — more images of one class than the other, which
  biases predictions toward the majority class.
- **Data noise** — corrupted / unreadable image files, plus detail loss
  from resizing every image to 128×128.

## The approach

| Step | Technique | Purpose |
|------|-----------|---------|
| Clean | Remove corrupted files | Train only on valid images |
| Balance | `class_weight='balanced'` | Treat both classes fairly |
| Augment | rotation, zoom, flip, shift | More variety, less overfitting |
| Improve | ResNet50 transfer learning | Stronger features than a small CNN |

A baseline CNN (trained from scratch) is compared against a ResNet50
transfer-learning model trained with class weights.

---

## How it works

```
clean corrupted images
        │
        ▼
augmented generators ──► Baseline CNN ──► evaluate ──► confusion matrix
        │
        └──────────────► ResNet50 + class weights ──► evaluate ──► confusion matrix
```

Two correctness details that matter:

- **Validation generator uses `shuffle=False`.** Otherwise the prediction
  order does not match `generator.classes`, and the confusion matrix
  compares mismatched rows — a common and silent bug.
- **ResNet50 gets its own `preprocess_input`** (not plain `1/255` scaling),
  and a `GlobalAveragePooling2D` head instead of `Flatten`, which keeps the
  trainable head small (~0.5 M params) and reduces overfitting.

---

## Project structure

```
.
├── main.py                 # full pipeline: clean → train → evaluate
├── requirements.txt
├── data/                   # dataset (not committed — see below)
├── results/                # generated confusion matrices
└── src/
    ├── data.py            # cleaning, generators, class weights
    ├── model.py            # baseline CNN + ResNet50 builders
    └── evaluate.py         # aligned predictions, metrics, plots
```

---

## Getting started

The dataset is **not included** (image folders are large). Place it as:

```
data/dataset/
    training_set/{cats, dogs}/
    test_set/{cats, dogs}/
```

Then:

```bash
pip install -r requirements.txt
python main.py
```

On Google Colab, set the dataset location with an environment variable:

```python
import os
os.environ["DATA_ROOT"] = "/content/dataset"
```

Training is GPU-friendly; Colab with a GPU runtime is recommended.

---

## Tech stack

`Python` · `TensorFlow / Keras` · `ResNet50` · `scikit-learn` · `Pillow` ·
`NumPy` · `Matplotlib`
