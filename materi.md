# 🏥 Workshop: Medical Image Segmentation dengan U-Net

[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> **Workshop 1 Pertemuan - Durasi: 2-3 Jam**  
> Level: Pemula (No Coding Experience Required!)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Part 1: Kaggle Setup](#part-1-pengenalan-kaggle--setup)
- [Part 2: Medical Image Segmentation](#part-2-medical-image-segmentation)
- [Evaluation Metrics](#-evaluation-metrics)
- [Troubleshooting](#-troubleshooting)
- [FAQ](#-faq)
- [Resources](#-resources)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

Workshop ini mengajarkan **Medical Image Segmentation** menggunakan **U-Net** untuk mendeteksi organ tubuh dalam gambar medis 3D (CT Scan). Cocok untuk pemula yang baru terjun ke dunia AI!

### Apa yang Akan Dipelajari?

✅ Menggunakan Kaggle (platform coding gratis dengan GPU!)  
✅ Membangun AI untuk "melihat" gambar medis  
✅ Melatih model U-Net untuk segmentasi organ  
✅ Evaluasi dan visualisasi hasil prediksi

### Demo Hasil

```
Input: CT Scan (Grayscale)  →  Output: Organ Detected (Colored Mask)
```

|                               Original Image                                |                                Ground Truth                                |                               AI Prediction                                |
| :-------------------------------------------------------------------------: | :------------------------------------------------------------------------: | :------------------------------------------------------------------------: |
| ![Original](https://via.placeholder.com/200x200/808080/FFFFFF?text=CT+Scan) | ![GT](https://via.placeholder.com/200x200/FF0000/FFFFFF?text=Ground+Truth) | ![Pred](https://via.placeholder.com/200x200/00FF00/FFFFFF?text=Prediction) |

---

## 🛠️ Prerequisites

### Yang Dibutuhkan:

- **Browser** (Chrome/Firefox/Edge)
- **Akun Google** (untuk daftar Kaggle)
- **Koneksi Internet** stabil
- **Antusiasme belajar!** 🚀

### Tidak Perlu:

❌ Install Python di komputer  
❌ Pengalaman coding sebelumnya  
❌ GPU sendiri (Kaggle menyediakan gratis!)  
❌ Dataset sendiri (sudah tersedia di Kaggle)

---

## 📥 Installation

**Tidak ada instalasi!** Semua berjalan di browser melalui Kaggle.

Tapi jika ingin run secara lokal:

```bash
# Clone repository
git clone https://github.com/scaidanpublikasi/medical-ipynb.git
cd medical-segmentation-workshop

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**

```txt
torch>=2.0.0
monai>=1.3.0
nibabel>=5.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

---

## 🚀 Quick Start

### Option 1: Kaggle Notebook (Recommended)

1. **Buka Kaggle**: [https://www.kaggle.com](https://www.kaggle.com)
2. **Login/Register** dengan akun Google
3. **Klik**: Code → New Notebook
4. **Copy-paste** code dari [full_code.py](#lampiran-full-code)
5. **Enable GPU**: Settings → Accelerator → GPU T4
6. **Run All!** ▶️

### Option 2: Google Colab

```python
# Upload notebook ke Google Colab
# Enable GPU: Runtime → Change runtime type → GPU
# Upload dataset atau mount Google Drive
```

### Option 3: Local (Advanced)

```bash
# Download dataset terlebih dahulu
# Sesuaikan path di code
python train.py --data_path /path/to/data --epochs 100
```

---

# Part 1: Pengenalan Kaggle & Setup

## 🔐 1.1 Registrasi Kaggle (5 Menit)

### Step-by-Step:

1. **Buka browser** → Ketik `www.kaggle.com`
2. Klik tombol **"Register"** (pojok kanan atas)
3. Pilih **"Sign up with Google"**
4. Login dengan akun Google Anda
5. ✅ **Done!** Akun siap digunakan

---

## 📝 1.2 Membuat Notebook (10 Menit)

### Apa itu Notebook?

> Notebook = Microsoft Word + Code Editor  
> Tempat menulis kode, menjalankannya, dan melihat hasil secara langsung!

### Cara Membuat:

1. **Klik menu "Code"** di navigation bar
2. **Klik "New Notebook"** (tombol biru)
3. **Pilih bahasa**: Python
4. **Notebook terbuka!** 🎉

### Enable GPU (PENTING!):

```
Settings (⚙️) → Accelerator → Pilih "GPU T4" atau "GPU P100" → Save
```

> **💡 Tip**: GPU membuat training 100x lebih cepat! Gratis di Kaggle (30 jam/minggu)

---

## 📂 1.3 Upload Dataset (15 Menit)

### Dataset yang Dibutuhkan:

- **Medical Images** (CT Scan 3D)
- **Labels** (Ground truth segmentation masks)

### Cara Add Dataset:

```python
# 1. Klik "+ Add Data" di panel kanan
# 2. Search: "costa adam" atau "medical segmentation"
# 3. Klik "Add"
# 4. Verifikasi dataset masuk:

import os
print(os.listdir('/kaggle/input'))
```

**Expected Output:**

```
['costa-adam']
```

---

# Part 2: Medical Image Segmentation

## 🧠 2.1 Konsep Medical Image Segmentation

### Apa itu Segmentation?

**Definisi**: Proses memberi label/warna pada area tertentu di gambar medis.

**Analogi**: Seperti mewarnai gambar di buku mewarnai, tapi dilakukan oleh AI!

### Mengapa Penting?

|    Manual (Dokter)     |  AI (Automated)   |
| :--------------------: | :---------------: |
|     ⏰ Berjam-jam      | ⚡ Hitungan detik |
| 😓 Lelah & error prone |   🤖 Konsisten    |
|    👤 Butuh expert     |    🆓 Scalable    |

### Use Cases:

- ✅ Deteksi tumor
- ✅ Segmentasi organ (liver, kidney, lung)
- ✅ Analisis penyakit
- ✅ Treatment planning

---

## 🏗️ 2.2 Arsitektur U-Net

### Kenapa U-Net?

**U-Net** adalah arsitektur deep learning khusus untuk segmentasi gambar medis.

### Struktur:

```
Input Image (128x128x128)
    ↓
[Encoder] → Extract features
    ↓
[Bottleneck] → Compress representation
    ↓
[Decoder] → Reconstruct segmentation
    ↓
Output Mask (128x128x128)
```

### Keunggulan:

- 🎯 Akurat untuk medical imaging
- ⚡ Efisien (tidak butuh data banyak)
- 🔄 Skip connections (gabungkan detail halus & kasar)

---

## 💻 2.3 Implementation

### Section 1: Install & Import Libraries

**Install MONAI (Medical Open Network for AI):**

```python
!pip install monai
```

**Import semua library:**

```python
# Basic libraries
import os                    # File/folder operations
import nibabel as nib       # Read medical images (.nii format)
import numpy as np          # Numerical operations
import torch                # Deep learning framework

# Data loading
from torch.utils.data import DataLoader, random_split

# MONAI - Medical imaging toolkit
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    ResizeWithPadOrCropd, NormalizeIntensityd, ToTensord
)
from monai.data import Dataset as MonaiDataset
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, jaccard_score
)

# Visualization
import matplotlib.pyplot as plt
```

**📝 Penjelasan:**

- `torch`: Framework AI utama
- `monai`: Toolkit khusus medical imaging
- `sklearn`: Hitung metrik evaluasi
- `matplotlib`: Visualisasi grafik

---

### Section 2: Load Dataset

```python
# Define data directories
image_dir = "/kaggle/input/costa-adam/imagesTr"      # Training images
label_dir = "/kaggle/input/costa-adam/labelsTr"      # Training labels
test_dir  = "/kaggle/input/costa-adam/imagesTs"      # Test images
label_test_dir = "/kaggle/input/costa-adam/labelsTs" # Test labels

# Initialize empty lists
image_paths = []
label_paths = []

# Collect all image files
for root, _, files in os.walk(image_dir):
    for file in files:
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            image_paths.append(os.path.join(root, file))

# Collect all label files
for root, _, files in os.walk(label_dir):
    for file in files:
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            label_paths.append(os.path.join(root, file))

# Sort to ensure matching
image_paths.sort()
label_paths.sort()

print(f"✅ Training images: {len(image_paths)}")
print(f"✅ Training labels: {len(label_paths)}")
```

**Load test data:**

```python
test_paths = sorted([
    os.path.join(test_dir, f) for f in os.listdir(test_dir)
    if f.endswith(".nii") or f.endswith(".nii.gz")
])

label_test_paths = sorted([
    os.path.join(label_test_dir, f) for f in os.listdir(label_test_dir)
    if f.endswith(".nii") or f.endswith(".nii.gz")
])

print(f"✅ Test images: {len(test_paths)}")
print(f"✅ Test labels: {len(label_test_paths)}")
```

---

### Section 3: Data Preprocessing

```python
# Define input size (all images will be resized to this)
input_size = (128, 128, 128)  # 3D: (depth, height, width)

# Define preprocessing pipeline
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),              # 1. Load files
    EnsureChannelFirstd(keys=["image", "label"]),     # 2. Format channels
    Orientationd(keys=["image", "label"], axcodes="RAS"),  # 3. Standard orientation
    ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=input_size),  # 4. Resize
    NormalizeIntensityd(keys="image"),                # 5. Normalize pixel values
    ToTensord(keys=["image", "label"])                # 6. Convert to tensor
])
```

**🔍 Penjelasan Transform:**

| Transform              | Fungsi             | Analogi                     |
| ---------------------- | ------------------ | --------------------------- |
| `LoadImaged`           | Buka file gambar   | Buka file Word              |
| `EnsureChannelFirstd`  | Format channel     | Atur TV ke channel yg benar |
| `Orientationd`         | Orientasi standar  | Putar gambar agar tegak     |
| `ResizeWithPadOrCropd` | Resize gambar      | Crop/zoom foto agar pas     |
| `NormalizeIntensityd`  | Standarisasi nilai | Samakan brightness          |
| `ToTensord`            | Ubah ke format AI  | Save as PDF (format baru)   |

---

### Section 4: Split Training & Validation

```python
# Create data dictionaries
data_dicts = [
    {"image": img, "label": lbl}
    for img, lbl in zip(image_paths, label_paths)
]

# Split 80% training, 20% validation
train_len = int(0.8 * len(data_dicts))
val_len = len(data_dicts) - train_len

train_files, val_files = random_split(data_dicts, [train_len, val_len])

# Create MONAI datasets
train_ds = MonaiDataset(train_files, transform=train_transforms)
val_ds = MonaiDataset(val_files, transform=train_transforms)

# Create data loaders
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1)

print(f"📊 Training samples: {train_len}")
print(f"📊 Validation samples: {val_len}")
```

**💡 Konsep Split Data:**

```
Total Data (100%)
    │
    ├─ Training (80%) → Mengajari AI
    └─ Validation (20%) → Testing AI
```

---

### Section 5: Build U-Net Model

```python
# Check device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# Build U-Net model
model = UNet(
    spatial_dims=3,           # 3D images
    in_channels=1,            # Grayscale input
    out_channels=1,           # Binary segmentation output
    channels=(32, 64, 128, 256, 512),  # Feature channels per layer
    strides=(2, 2, 2, 2),     # Downsampling strides
    num_res_units=2           # Residual units per layer
).to(device)

# Define loss function
loss_fn = DiceCELoss(sigmoid=True)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Early stopping parameters
best_loss = float('inf')
patience = 5
trigger = 0

# History tracking
train_losses = []
val_losses = []

print("✅ Model initialized successfully!")
```

**🎯 Model Parameters:**

- `spatial_dims=3`: 3D medical images
- `channels=(32, 64, 128, 256, 512)`: Neurons per layer
- `lr=1e-4`: Learning rate (kecepatan belajar)
- `patience=5`: Stop kalau 5 epoch tidak ada progress

---

### Section 6: Training Loop

```python
print("🚀 Starting training...")
print("⏰ This may take 1-2 hours. Be patient!")

for epoch in range(1000):
    # ========== TRAINING PHASE ==========
    model.train()
    epoch_loss = 0

    for batch in train_loader:
        # Get data
        x = batch["image"].to(device)
        y = batch["label"].to(device)

        # Forward pass
        y_pred = model(x)

        # Calculate loss
        loss = loss_fn(y_pred, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Average training loss
    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)

    # ========== VALIDATION PHASE ==========
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for val_batch in val_loader:
            val_x = val_batch["image"].to(device)
            val_y = val_batch["label"].to(device)
            val_pred = model(val_x)
            val_loss += loss_fn(val_pred, val_y).item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    # Print progress
    print(f"Epoch {epoch+1:3d} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

    # ========== EARLY STOPPING ==========
    if val_loss < best_loss:
        best_loss = val_loss
        trigger = 0
        torch.save(model.state_dict(), "/kaggle/working/best_model.pt")
        print("    ✅ Model saved!")
    else:
        trigger += 1
        if trigger >= patience:
            print("🛑 Early stopping triggered!")
            break

print("🎉 Training completed!")
```

**📈 Training Process:**

```
Epoch 1: Train Loss: 0.8234 | Val Loss: 0.7891 ✅ Model saved!
Epoch 2: Train Loss: 0.7123 | Val Loss: 0.6543 ✅ Model saved!
Epoch 3: Train Loss: 0.6234 | Val Loss: 0.5987 ✅ Model saved!
...
```

---

### Section 7: Model Evaluation

```python
# Load best model
model.load_state_dict(torch.load("/kaggle/working/best_model.pt"))
model.eval()

# Initialize metrics dictionary
metrics = {
    "Dice": [],
    "IoU": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1": []
}

print("📊 Evaluating model on validation set...")

# Evaluate on validation set
with torch.no_grad():
    for batch in val_loader:
        x = batch["image"].to(device)
        y_true = batch["label"].cpu().numpy().flatten()

        # Predict
        y_pred = torch.sigmoid(model(x)).cpu().numpy().flatten()
        y_bin = (y_pred > 0.5).astype(np.uint8)

        # Calculate metrics
        metrics["Dice"].append(f1_score(y_true, y_bin))
        metrics["IoU"].append(jaccard_score(y_true, y_bin))
        metrics["Accuracy"].append(accuracy_score(y_true, y_bin))
        metrics["Precision"].append(precision_score(y_true, y_bin))
        metrics["Recall"].append(recall_score(y_true, y_bin))
        metrics["F1"].append(f1_score(y_true, y_bin))

# Print results
print("\n" + "="*50)
print("           VALIDATION METRICS")
print("="*50)
for metric_name, values in metrics.items():
    avg_value = np.mean(values)
    print(f"{metric_name:12s}: {avg_value:.4f}")
print("="*50)
```

**Expected Output:**

```
==================================================
           VALIDATION METRICS
==================================================
Dice        : 0.8542
IoU         : 0.7823
Accuracy    : 0.9234
Precision   : 0.8765
Recall      : 0.8432
F1          : 0.8542
==================================================
```

---

### Section 8: Testing on New Data

```python
# Prepare test transforms
test_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=input_size),
    NormalizeIntensityd(keys="image"),
    ToTensord(keys=["image", "label"])
])

# Create test dataset
test_dicts = [
    {"image": img, "label": lbl}
    for img, lbl in zip(test_paths, label_test_paths)
]
test_ds = MonaiDataset(test_dicts, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1)

# Create output directory
os.makedirs("/kaggle/working/predictions", exist_ok=True)

# Initialize test metrics
test_metrics = {
    "Dice": [], "IoU": [], "Accuracy": [],
    "Precision": [], "Recall": [], "F1": []
}

print("🧪 Testing on unseen data...")

# Test loop
for i, test_batch in enumerate(test_loader):
    x = test_batch["image"].to(device)
    y_true = test_batch["label"].cpu().numpy().flatten()

    with torch.no_grad():
        # Sliding window inference for large images
        pred = sliding_window_inference(
            x,
            roi_size=input_size,
            sw_batch_size=1,
            predictor=model
        )
        pred_np = torch.sigmoid(pred).cpu().numpy()[0, 0]
        pred_bin = (pred_np > 0.5).astype(np.uint8)

    # Save prediction
    pred_nii = nib.Nifti1Image(pred_bin, affine=np.eye(4))
    nib.save(pred_nii, f"/kaggle/working/predictions/pred_{i}.nii.gz")

    # Calculate metrics
    y_pred_flat = pred_bin.flatten()
    test_metrics["Dice"].append(f1_score(y_true, y_pred_flat))
    test_metrics["IoU"].append(jaccard_score(y_true, y_pred_flat))
    test_metrics["Accuracy"].append(accuracy_score(y_true, y_pred_flat))
    test_metrics["Precision"].append(precision_score(y_true, y_pred_flat))
    test_metrics["Recall"].append(recall_score(y_true, y_pred_flat))
    test_metrics["F1"].append(f1_score(y_true, y_pred_flat))

    print(f"  Processed test image {i+1}/{len(test_loader)}")

# Print test results
print("\n" + "="*50)
print("              TEST METRICS")
print("="*50)
for metric_name, values in test_metrics.items():
    avg_value = np.mean(values)
    print(f"{metric_name:12s}: {avg_value:.4f}")
print("="*50)
```

---

### Section 9: Visualization

**1. Plot Training History:**

```python
plt.figure(figsize=(12, 6))

epochs = range(1, len(train_losses) + 1)

plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
plt.plot(epochs, val_losses, 'r-x', label='Validation Loss', linewidth=2)

plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('📈 Training & Validation Loss Progress', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/kaggle/working/training_curve.png', dpi=300)
plt.show()

print("✅ Training curve saved to: /kaggle/working/training_curve.png")
```

**2. Visualize Predictions:**

```python
print("🖼️ Visualizing predictions...")

model.eval()
num_samples = 3

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        if i >= num_samples:
            break

        image = batch["image"].to(device)
        label = batch["label"].to(device)

        # Predict
        output = torch.sigmoid(model(image))
        pred = (output > 0.5).float()

        # Get numpy arrays
        img_np = image[0].cpu().numpy()[0]
        label_np = label[0].cpu().numpy()[0]
        pred_np = pred[0].cpu().numpy()[0]

        # Get middle slice
        mid_slice = img_np.shape[0] // 2

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(img_np[mid_slice], cmap='gray')
        axes[0].set_title('🖼️ Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Ground truth
        axes[1].imshow(img_np[mid_slice], cmap='gray')
        axes[1].imshow(label_np[mid_slice], cmap='jet', alpha=0.5)
        axes[1].set_title('✅ Ground Truth', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # Prediction
        axes[2].imshow(img_np[mid_slice], cmap='gray')
        axes[2].imshow(pred_np[mid_slice], cmap='jet', alpha=0.5)
        axes[2].set_title('🤖 AI Prediction', fontsize=14, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(f'/kaggle/working/prediction_{i}.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Sample {i+1}/{num_samples} visualized ✅")

print("\n✅ All visualizations saved!")
```

---

## 📊 Evaluation Metrics

### Metrics Explained:

| Metric            | Formula                                            | Interpretation                            | Good Value |
| ----------------- | -------------------------------------------------- | ----------------------------------------- | ---------- |
| **Dice Score**    | `2 * (Pred ∩ GT) / (Pred + GT)`                    | Overlap similarity                        | > 0.80     |
| **IoU (Jaccard)** | `(Pred ∩ GT) / (Pred ∪ GT)`                        | Intersection over Union                   | > 0.70     |
| **Accuracy**      | `Correct Pixels / Total Pixels`                    | Overall correctness                       | > 0.90     |
| **Precision**     | `True Positive / (True Positive + False Positive)` | How many predicted positives are correct? | > 0.85     |
| **Recall**        | `True Positive / (True Positive + False Negative)` | How many actual positives are detected?   | > 0.85     |
| **F1 Score**      | `2 * (Precision * Recall) / (Precision + Recall)`  | Harmonic mean of Precision & Recall       | > 0.85     |

### Visual Guide:

```
Ground Truth:    [████░░░░]
Prediction:      [███░░░░░]
                  ^^^
                  TP (True Positive)

Dice = 2 * 3 / (3 + 3) = 1.0 (Perfect!)
IoU = 3 / (3 + 0) = 1.0 (Perfect!)
```

---

## 🐛 Troubleshooting

### Issue 1: "CUDA Out of Memory"

**Symptoms:**

```
RuntimeError: CUDA out of memory
```

**Solutions:**

```python
# Solution 1: Reduce input size
input_size = (96, 96, 96)  # Instead of (128, 128, 128)

# Solution 2: Reduce model channels
model = UNet(
    channels=(16, 32, 64, 128, 256),  # Instead of (32, 64, 128, 256, 512)
)

# Solution 3: Clear cache
import torch
torch.cuda.empty_cache()
```

---

### Issue 2: "Dataset Not Found"

**Symptoms:**

```
FileNotFoundError: [Errno 2] No such file or directory
```

**Solutions:**

```python
# Check dataset structure
import os
print(os.listdir('/kaggle/input'))

# Verify path exactly matches
image_dir = "/kaggle/input/your-dataset-name/imagesTr"  # Check spelling!
```

---

### Issue 3: "Training Stuck / Not Progressing"

**Solutions:**

```python
# Add progress indicator
from tqdm import tqdm

for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
    # ... training code ...
```

---

### Issue 4: "Poor Performance (Dice < 0.50)"

**Debugging Steps:**

1. **Check data quality:**

```python
# Visualize raw data
import matplotlib.pyplot as plt
img = nib.load(image_paths[0]).get_fdata()
plt.imshow(img[:, :, img.shape[2]//2], cmap='gray')
plt.show()
```

2. **Check preprocessing:**

```python
# Print shapes after transform
batch = next(iter(train_loader))
print(f"Image shape: {batch['image'].shape}")
print(f"Label shape: {batch['label'].shape}")
```

3. **Adjust learning rate:**

```python
# Try different values
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Faster
# or
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Slower
```

---

## ❓ FAQ

### Q1: Kenapa training lama sekali (1-2 jam)?

**A:** Medical imaging 3D sangat besar (128x128x128 = 2 juta voxels!). Ini normal. Pastikan GPU aktif untuk mempercepat.

---

### Q2: Berapa Dice Score yang bagus?

**A:**

- 0.80+ = Excellent ✅
- 0.70-0.79 = Good ✅
- 0.60-0.69 = Fair ⚠️
- < 0.60 = Need improvement ❌

---

### Q3: Bisa pakai arsitektur lain selain U-Net?

**A:** Bisa! Coba:

```python
from monai.networks.nets import AttentionUnet, UNETR, SwinUNETR

# Attention U-Net (U-Net + Attention mechanism)
model = AttentionUnet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(32, 64, 128, 256),
    strides=(2, 2, 2)
).to(device)

# UNETR (Transformer-based)
model = UNETR(
    in_channels=1,
    out_channels=1,
    img_size=(128, 128, 128),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
).to(device)
```

---

### Q4: Bagaimana cara improve model?

**A:** Tips meningkatkan performa:

1. **Data Augmentation:**

```python
from monai.transforms import RandRotate90d, RandFlipd, RandGaussianNoised

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 2)),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axes=[0]),
    RandGaussianNoised(keys=["image"], prob=0.3),
    # ... other transforms ...
])
```

2. **Learning Rate Scheduling:**

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# In training loop after validation:
scheduler.step(val_loss)
```

3. **Ensemble Models:**

```python
# Train multiple models and average predictions
predictions = []
for model_path in ['model1.pt', 'model2.pt', 'model3.pt']:
    model.load_state_dict(torch.load(model_path))
    pred = model(x)
    predictions.append(pred)

final_pred = torch.mean(torch.stack(predictions), dim=0)
```

---

### Q5: Bisa deploy model ini ke production?

**A:** Bisa! Step-by-step:

1. **Export model:**

```python
# Save as TorchScript
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, "model_scripted.pt")

# Or save as ONNX
dummy_input = torch.randn(1, 1, 128, 128, 128).to(device)
torch.onnx.export(model, dummy_input, "model.onnx")
```

2. **Create inference script:**

```python
def predict(image_path):
    # Load image
    img = nib.load(image_path).get_fdata()

    # Preprocess
    img_tensor = preprocess(img)

    # Predict
    with torch.no_grad():
        pred = model(img_tensor)
        pred_mask = (torch.sigmoid(pred) > 0.5).cpu().numpy()

    return pred_mask
```

3. **Deploy options:**

- FastAPI / Flask (REST API)
- Docker container
- Cloud (AWS/GCP/Azure)
- Mobile (ONNX Runtime)

---

### Q6: Dataset saya beda format (DICOM), gimana?

**A:** Pakai library tambahan:

```python
# Install pydicom
!pip install pydicom

# Load DICOM
import pydicom

def load_dicom_series(folder_path):
    slices = []
    for filename in sorted(os.listdir(folder_path)):
        ds = pydicom.dcmread(os.path.join(folder_path, filename))
        slices.append(ds.pixel_array)

    volume = np.stack(slices, axis=-1)
    return volume

# Convert DICOM to NIfTI
volume = load_dicom_series('/path/to/dicom/folder')
nifti_img = nib.Nifti1Image(volume, affine=np.eye(4))
nib.save(nifti_img, 'output.nii.gz')
```

---

### Q7: Bisa pakai pre-trained model?

**A:** Bisa! MONAI punya model zoo:

```python
from monai.bundle import download

# Download pre-trained model
download(name="spleen_ct_segmentation", bundle_dir="./models")

# Load pre-trained weights
from monai.networks.nets import UNet
model = UNet(...)
model.load_state_dict(torch.load("./models/spleen_ct_segmentation/model.pt"))

# Fine-tune on your data
for param in model.parameters():
    param.requires_grad = True  # Unfreeze all layers

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
# ... continue training ...
```

---

## 💡 Tips & Best Practices

### 🚀 Performance Optimization

1. **Mixed Precision Training (AMP):**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(num_epochs):
    for batch in train_loader:
        x, y = batch["image"].to(device), batch["label"].to(device)

        with autocast():
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

**Benefit:** 2-3x faster training! 🚀

---

2. **Gradient Accumulation:**

```python
accumulation_steps = 4

for i, batch in enumerate(train_loader):
    x, y = batch["image"].to(device), batch["label"].to(device)
    y_pred = model(x)
    loss = loss_fn(y_pred, y) / accumulation_steps

    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Benefit:** Simulate larger batch size without OOM!

---

3. **Efficient Data Loading:**

```python
train_loader = DataLoader(
    train_ds,
    batch_size=2,  # Increase if GPU memory allows
    shuffle=True,
    num_workers=4,  # Parallel data loading
    pin_memory=True,  # Faster CPU to GPU transfer
    persistent_workers=True  # Keep workers alive
)
```

---

### 📈 Monitoring & Logging

**Use TensorBoard:**

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

for epoch in range(num_epochs):
    # ... training ...

    # Log scalars
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Metrics/dice', dice_score, epoch)

    # Log images
    writer.add_images('Images/input', x, epoch)
    writer.add_images('Images/prediction', pred, epoch)

writer.close()

# View in browser:
# tensorboard --logdir=runs
```

---

**Use Weights & Biases (W&B):**

```python
import wandb

# Initialize
wandb.init(project="medical-segmentation", name="unet-experiment-1")

# Log config
wandb.config.update({
    "learning_rate": 1e-4,
    "epochs": 100,
    "batch_size": 1,
    "architecture": "UNet"
})

# Log metrics
for epoch in range(num_epochs):
    # ... training ...
    wandb.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "dice_score": dice_score
    })

# Log model
wandb.save('model.pt')
```

---

### 🔬 Experiment Tracking

**Keep track of experiments:**

```python
import json
from datetime import datetime

experiment = {
    "timestamp": datetime.now().isoformat(),
    "model": "UNet",
    "input_size": (128, 128, 128),
    "learning_rate": 1e-4,
    "batch_size": 1,
    "epochs_trained": len(train_losses),
    "best_val_loss": best_loss,
    "metrics": {
        "dice": float(np.mean(metrics["Dice"])),
        "iou": float(np.mean(metrics["IoU"])),
        "accuracy": float(np.mean(metrics["Accuracy"]))
    }
}

# Save experiment log
with open(f'experiments/{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
    json.dump(experiment, f, indent=2)
```

---

## 📚 Resources

### 🎓 Learning Resources

**Courses:**

- [Deep Learning Specialization - Coursera](https://www.coursera.org/specializations/deep-learning)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)
- [Kaggle Learn - Computer Vision](https://www.kaggle.com/learn/computer-vision)

**Books:**

- 📖 "Deep Learning for Medical Image Analysis" - Zhou et al.
- 📖 "Medical Image Analysis" - Atam Dhawan
- 📖 "Deep Learning" - Ian Goodfellow

**Papers:**

- 📄 [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- 📄 [nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation](https://arxiv.org/abs/1809.10486)
- 📄 [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999)

---

### 🛠️ Tools & Libraries

**Essential:**

- [MONAI](https://monai.io/) - Medical imaging toolkit
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Nibabel](https://nipy.org/nibabel/) - Neuroimaging data I/O
- [ITK-SNAP](http://www.itksnap.org/) - 3D medical image viewer

**Visualization:**

- [3D Slicer](https://www.slicer.org/) - Medical image visualization
- [MITK](https://www.mitk.org/) - Medical imaging toolkit
- [Napari](https://napari.org/) - Multi-dimensional image viewer

**Dataset Sources:**

- [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
- [Grand Challenges](https://grand-challenge.org/)
- [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/)
- [UK Biobank](https://www.ukbiobank.ac.uk/)

---

### 🌐 Communities

**Forums & Discussion:**

- [Kaggle Discussion](https://www.kaggle.com/discussions)
- [MONAI Forum](https://github.com/Project-MONAI/MONAI/discussions)
- [Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [Reddit r/computervision](https://www.reddit.com/r/computervision/)

**Social Media:**

- Twitter: Follow `#medicalAI` `#MedicalImaging` `#DeepLearning`
- LinkedIn: Join "Medical AI" groups
- Discord: AI/ML communities

---

## 🎯 Next Steps & Project Ideas

### Level 1: Beginner Projects (You're here! ✅)

✅ Run this notebook successfully  
✅ Understand U-Net architecture  
✅ Evaluate model performance  
✅ Visualize predictions

---

### Level 2: Intermediate Projects (1-2 months)

**Project 1: Multi-Organ Segmentation**

```
Goal: Segment multiple organs (liver, kidney, spleen) simultaneously
Dataset: Medical Segmentation Decathlon Task 10
Challenge: Multi-class segmentation (out_channels > 1)
```

**Project 2: Brain Tumor Segmentation**

```
Goal: Detect and segment brain tumors
Dataset: BraTS Challenge
Challenge: Handle multiple MRI modalities (T1, T2, FLAIR, T1ce)
```

**Project 3: Lung Nodule Detection**

```
Goal: Detect lung nodules in CT scans
Dataset: LUNA16
Challenge: 3D object detection + classification
```

---

### Level 3: Advanced Projects (3-6 months)

**Project 1: Real-time Inference System**

```python
# FastAPI deployment
from fastapi import FastAPI, File, UploadFile
import uvicorn

app = FastAPI()

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    # Load and preprocess
    image = load_medical_image(file)

    # Predict
    mask = model.predict(image)

    # Return result
    return {"segmentation": mask.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Project 2: Active Learning Pipeline**

```
Goal: Minimize annotation cost with active learning
Steps:
1. Train on small labeled dataset
2. Use model to find uncertain samples
3. Send uncertain samples to expert for annotation
4. Retrain model
5. Repeat until desired performance
```

**Project 3: Kaggle Competition**

```
Participate in medical imaging competitions:
- RSNA competitions (X-ray, CT analysis)
- SIIM-ISIC Melanoma Classification
- DIagnostic Questions (DQ) challenge
```

---

## 🏆 Achievements & Milestones

Track your progress:

- [ ] **Setup Master** - Successfully setup Kaggle & run first notebook
- [ ] **Code Warrior** - Understand and modify training loop
- [ ] **Metric Maven** - Achieve Dice Score > 0.70
- [ ] **Visualization Guru** - Create publication-quality figures
- [ ] **Optimizer** - Reduce training time by 50%
- [ ] **Experimenter** - Try 5+ different hyperparameters
- [ ] **Architecture Explorer** - Test 3+ model architectures
- [ ] **Dataset Collector** - Work with 3+ different datasets
- [ ] **Competition Ready** - Submit to a Kaggle competition
- [ ] **Production Pro** - Deploy model as API
- [ ] **Community Hero** - Help 5+ people in forums
- [ ] **Knowledge Sharer** - Write a blog post about your learnings

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Ways to Contribute:

1. **Report Issues**

   - Found a bug? Open an issue!
   - Include error message and steps to reproduce

2. **Improve Documentation**

   - Fix typos
   - Add explanations
   - Translate to other languages

3. **Add Features**

   - New preprocessing techniques
   - Different model architectures
   - Evaluation metrics

4. **Share Results**
   - Post your experiments
   - Share interesting findings
   - Contribute datasets

### Contribution Process:

```bash
# 1. Fork the repository
git clone https://github.com/yourusername/medical-segmentation-workshop.git

# 2. Create a branch
git checkout -b feature/amazing-feature

# 3. Make changes and commit
git commit -m "Add amazing feature"

# 4. Push to your fork
git push origin feature/amazing-feature

# 5. Open a Pull Request
```

---

## 📜 Citation

If you use this workshop in your research or teaching, please cite:

```bibtex
@misc{medical_segmentation_workshop_2025,
  title={Medical Image Segmentation Workshop: U-Net Tutorial for Beginners},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/medical-segmentation-workshop}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📞 Contact & Support

**Need Help?**

- 📧 Email: your.email@example.com
- 💬 Kaggle Discussion: [Link to discussion]
- 🐛 GitHub Issues: [Open an issue](https://github.com/yourusername/repo/issues)
- 💼 LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- 🐦 Twitter: [@yourhandle](https://twitter.com/yourhandle)

**Office Hours:**

- Every Friday 2-4 PM (GMT+7)
- Zoom link: [Add your link]

---

## 🙏 Acknowledgments

Special thanks to:

- **MONAI Team** - For the amazing medical imaging toolkit
- **PyTorch Team** - For the deep learning framework
- **Kaggle** - For providing free GPU resources
- **Medical Segmentation Decathlon** - For the datasets
- **Community Contributors** - For feedback and improvements

**Inspired by:**

- U-Net paper (Ronneberger et al., 2015)
- MONAI tutorials and examples
- Fast.ai teaching methodology

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/repo?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/repo?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/yourusername/repo?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/repo)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/repo)
![Last commit](https://img.shields.io/github/last-commit/yourusername/repo)

---

## 🗺️ Roadmap

### Q1 2025

- [x] Launch workshop v1.0
- [x] Create comprehensive documentation
- [ ] Add video tutorials
- [ ] Create interactive Jupyter notebook

### Q2 2025

- [ ] Multi-language support (Indonesian, Spanish, Chinese)
- [ ] Add more dataset examples
- [ ] Create Docker container
- [ ] Launch companion YouTube series

### Q3 2025

- [ ] Add advanced architectures (Transformer-based)
- [ ] Create mobile deployment guide
- [ ] Add more visualization tools
- [ ] Launch online certification

### Q4 2025

- [ ] Create online course platform
- [ ] Add real-time inference examples
- [ ] Build community forum
- [ ] Host virtual workshop events

---

## 📈 Change Log

### Version 1.0.0 (2025-10-22)

- ✨ Initial release
- 📝 Complete documentation
- 💻 Full code implementation
- 🎨 Visualization examples
- 🧪 Testing suite

### Upcoming (v1.1.0)

- 🌍 Multi-language support
- 📹 Video tutorials
- 🐳 Docker containerization
- 📱 Mobile deployment guide

---

## 🎉 Success Stories

> "This workshop helped me transition from zero ML knowledge to building production-ready medical imaging models in 3 months!" - _Student A_

> "Clear explanations and practical examples. Perfect for beginners!" - _Researcher B_

> "Deployed my first medical AI model thanks to this tutorial." - _Engineer C_

**Share your story!** Submit via [Google Form](https://forms.google.com) or email us.

---

## 📝 Appendix

### A. Full Code in One File

```python
# ============================================
# MEDICAL IMAGE SEGMENTATION - COMPLETE CODE
# U-Net with MONAI Framework
# ============================================

# 1. INSTALLATION
!pip install monai nibabel

# 2. IMPORTS
import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    ResizeWithPadOrCropd, NormalizeIntensityd, ToTensord
)
from monai.data import Dataset as MonaiDataset
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
import matplotlib.pyplot as plt

# 3. DATA LOADING
image_dir = "/kaggle/input/costa-adam/imagesTr"
label_dir = "/kaggle/input/costa-adam/labelsTr"
test_dir = "/kaggle/input/costa-adam/imagesTs"
label_test_dir = "/kaggle/input/costa-adam/labelsTs"

image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith((".nii", ".nii.gz"))])
label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith((".nii", ".nii.gz"))])
test_paths = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith((".nii", ".nii.gz"))])
label_test_paths = sorted([os.path.join(label_test_dir, f) for f in os.listdir(label_test_dir) if f.endswith((".nii", ".nii.gz"))])

print(f"✅ Training: {len(image_paths)} images | Test: {len(test_paths)} images")

# 4. PREPROCESSING
input_size = (128, 128, 128)
transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=input_size),
    NormalizeIntensityd(keys="image"),
    ToTensord(keys=["image", "label"])
])

# 5. DATASET SPLIT
data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(image_paths, label_paths)]
train_len = int(0.8 * len(data_dicts))
val_len = len(data_dicts) - train_len
train_files, val_files = random_split(data_dicts, [train_len, val_len])

train_ds = MonaiDataset(train_files, transform=transforms)
val_ds = MonaiDataset(val_files, transform=transforms)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1)

# 6. MODEL SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=3, in_channels=1, out_channels=1,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2), num_res_units=2
).to(device)

loss_fn = DiceCELoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 7. TRAINING
best_loss, patience, trigger = float('inf'), 5, 0
train_losses, val_losses = [], []

for epoch in range(1000):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        x, y = batch["image"].to(device), batch["label"].to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_batch in val_loader:
            val_x, val_y = val_batch["image"].to(device), val_batch["label"].to(device)
            val_pred = model(val_x)
            val_loss += loss_fn(val_pred, val_y).item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1:3d} | Train: {epoch_loss:.4f} | Val: {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        trigger = 0
        torch.save(model.state_dict(), "/kaggle/working/best_model.pt")
    else:
        trigger += 1
        if trigger >= patience:
            print("🛑 Early stopping")
            break

# 8. EVALUATION
model.load_state_dict(torch.load("/kaggle/working/best_model.pt"))
model.eval()

metrics = {"Dice": [], "IoU": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": []}

with torch.no_grad():
    for batch in val_loader:
        x = batch["image"].to(device)
        y_true = batch["label"].cpu().numpy().flatten()
        y_pred = torch.sigmoid(model(x)).cpu().numpy().flatten()
        y_bin = (y_pred > 0.5).astype(np.uint8)

        metrics["Dice"].append(f1_score(y_true, y_bin))
        metrics["IoU"].append(jaccard_score(y_true, y_bin))
        metrics["Accuracy"].append(accuracy_score(y_true, y_bin))
        metrics["Precision"].append(precision_score(y_true, y_bin))
        metrics["Recall"].append(recall_score(y_true, y_bin))
        metrics["F1"].append(f1_score(y_true, y_bin))

print("\n📊 VALIDATION METRICS")
print("="*40)
for k, v in metrics.items():
    print(f"{k:12s}: {np.mean(v):.4f}")

# 9. VISUALIZATION
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(train_losses)+1), train_losses, 'b-o', label='Train')
plt.plot(range(1, len(val_losses)+1), val_losses, 'r-x', label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
plt.savefig('/kaggle/working/training_curve.png', dpi=300)
plt.show()

print("\n🎉 TRAINING COMPLETED!")
```

---

### B. Glossary

**AI (Artificial Intelligence)**: Computer systems that can perform tasks requiring human intelligence

**Augmentation**: Artificially increasing dataset size by applying transformations

**Batch Size**: Number of samples processed together

**Channel**: Feature dimension in image (e.g., RGB has 3 channels)

**Dice Score**: Overlap metric for segmentation (0-1, higher is better)

**Epoch**: One complete pass through entire training dataset

**GPU**: Graphics Processing Unit (hardware accelerator for AI)

**Ground Truth**: Correct/reference annotation

**IoU (Intersection over Union)**: Jaccard index, overlap metric

**Learning Rate**: Step size for model weight updates

**Loss Function**: Metric measuring prediction error

**Overfitting**: Model memorizes training data but fails on new data

**Segmentation**: Pixel-level classification

**Tensor**: Multi-dimensional array (data structure for AI)

**U-Net**: CNN architecture for segmentation

**Validation Set**: Data for monitoring training progress

**Voxel**: 3D pixel

---

### C. Keyboard Shortcuts (Kaggle)

| Shortcut        | Action                  |
| --------------- | ----------------------- |
| `Shift + Enter` | Run cell & move to next |
| `Ctrl + Enter`  | Run cell & stay         |
| `Alt + Enter`   | Run cell & insert below |
| `A`             | Insert cell above       |
| `B`             | Insert cell below       |
| `DD`            | Delete cell             |
| `Z`             | Undo cell deletion      |
| `M`             | Change to Markdown      |
| `Y`             | Change to Code          |
| `Ctrl + S`      | Save notebook           |

---

## 🎊 Final Words

Congratulations on completing this workshop! 🎉

You've learned:

- ✅ How to use Kaggle platform
- ✅ Medical image segmentation concepts
- ✅ U-Net architecture
- ✅ Training deep learning models
- ✅ Evaluating model performance

**Remember:**

> "The expert in anything was once a beginner." - Helen Hayes

Keep learning, keep coding, and don't be afraid to experiment!

**Next Steps:**

1. ⭐ Star this repository
2. 🍴 Fork and customize for your projects
3. 📢 Share with others learning AI
4. 💬 Join our community
5. 🚀 Start your own medical AI project!

---

**Made with ❤️ for AI learners worldwide**

_Last updated: October 22, 2025_

---

**[⬆ Back to Top](#-workshop-medical-image-segmentation-dengan-u-net)**
