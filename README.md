# 🚀 TransUNet for SAR Oil Spill Segmentation (SOS Dataset)

This repository contains the implementation of **TransUNet** adapted for **oil spill segmentation in Synthetic Aperture Radar (SAR) satellite imagery**. The model is trained and evaluated using the **SOS (SAR Oil Spill) dataset**, which incorporates imagery from both **Sentinel-1** and **ALOS-PALSAR** sensors.

---

## 🎯 Task Definition

The objective is **binary semantic segmentation** of oil spill regions within SAR images.

* **Oil Spill → White (1)**
* **Sea / Background → Black (0)**

---

## 🧠 Model Architecture: TransUNet

The architecture leverages the strengths of both Transformers and Convolutional Neural Networks:

* **Transformer Encoder (ViT Backbone):** Captures **global context** and long-range dependencies across the SAR scene, which is crucial for understanding large spill shapes.
* **U-Net Decoder:** Utilizes **skip connections** from the encoder to retain and refine **fine-grained spatial details** necessary for accurate boundary delineation.

<img width="1063" height="519" alt="image" src="https://github.com/user-attachments/assets/0e5287de-a6cd-4edc-a8c7-a1c06a703767" />

---

## ✅ Key Features

| Feature | Status | Description |
| :--- | :--- | :--- |
| **SAR Oil Spill Dataset Support** | ✅ | Ready to process the SOS dataset structure. |
| **Binary Segmentation** | ✅ | Output is a grayscale mask (0 for sea, 1 for oil). |
| **TransUNet Implementation** | ✅ | Uses a Vision Transformer backbone. |
| **Custom Pipeline** | ✅ | Includes necessary scripts for training/testing on this specific dataset. |
| **Prediction Visualization** | ✅ | Generates output images for easy review. |
| **Evaluation Metric** | ✅ | **Dice Score** is used as the primary performance metric. |

---

## 📂 Dataset Structure Requirement

Ensure your SOS dataset is organized as follows for the scripts to work correctly:

```bash
dataset/
├─ train/
│  ├─ palsar/
│  │  ├─ image/
│  │  └─ label/
│  └─ sentinel/
│     ├─ image/
│     └─ label/
└─ test/
   ├─ palsar/
   │  ├─ image/
   │  └─ label/
   └─ sentinel/
      ├─ image/
      └─ label/
```

---

## 🛠️ Setup & Execution

Follow these steps to set up the environment and run the model.

### 1. Environment Setup

Create and activate a dedicated virtual environment:

```bash
# 1. Create Virtual Environment
python3 -m venv transunet_env
source transunet_env/bin/activate   # Linux/Mac
# transunet_env\Scripts\activate    # Windows

# 2. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. PyTorch Installation (Crucial for GPU usage)

Install the correct PyTorch version matching your CUDA setup.

**For GPU (e.g., CUDA 11.8):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU Only:**

```bash
pip install torch torchvision torchaudio
```

---

### 3. Pre-trained Weights Download

Download the ImageNet-21k pre-trained weights for the Vision Transformer backbone:

```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/R50-ViT-B_16.npz
mkdir -p ../model/vit_checkpoint/imagenet21k
mv R50-ViT-B_16.npz ../model/vit_checkpoint/imagenet21k/
```

---

### 4. Prepare Data Lists

Generate the necessary `.txt` files that list the image paths for training, validation, and testing.

```bash
python create_lists.py
# This generates files like: lists/lists_OilSpill/train.txt, val.txt, test_vol.txt
```

---

## 🏋️ Training and Inference

### Training the Model

Start the training process using the specified dataset and ViT configuration.

```bash
python train.py --dataset OilSpill --vit_name R50-ViT-B_16
```

Model checkpoints will be saved automatically in: `model/TU_OilSpill224/`

---

### Testing / Inference

Run inference on your test set and save the resulting segmentations.

```bash
python test.py --dataset OilSpill --vit_name R50-ViT-B_16 --is_savenii
```

**Output Files** are saved in: `predictions/TU_OilSpill224/`

| File | Meaning |
| :--- | :--- |
| `*_prediction.png` | Predicted grayscale mask (White = Oil, Black = Sea) |
| `*_original.png` | Original SAR input image |

---

## 📊 Example Output Visualization

| Input SAR Image | Output Segmentation Mask |
| :--- | :--- |
| Grayscale SAR image | White oil spill regions sharply contrasted against the Black sea background. |

---

## 📈 Performance Metrics

The model is evaluated using the **Dice Similarity Coefficient (DSC)**, which measures the overlap between predicted and ground truth masks:

$$
\text{Dice} = \frac{2 \times |X \cap Y|}{|X| + |Y|}
$$

Where:
- $X$ = Predicted mask
- $Y$ = Ground truth mask

---
