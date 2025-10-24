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
- [Machine Learning Overview](#-machine-learning-overview)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Try Pre-trained Model](#-try-pre-trained-model-inference)
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
✅ **Menggunakan model yang sudah jadi untuk inference** ⭐ NEW!

### Demo Hasil

```
Input: CT Scan (Grayscale)  →  Output: Organ Detected (Colored Mask)
```

|                               Original Image                                |                                Ground Truth                                |                               AI Prediction                                |
| :-------------------------------------------------------------------------: | :------------------------------------------------------------------------: | :------------------------------------------------------------------------: |
| ![Original](https://via.placeholder.com/200x200/808080/FFFFFF?text=CT+Scan) | ![GT](https://via.placeholder.com/200x200/FF0000/FFFFFF?text=Ground+Truth) | ![Pred](https://via.placeholder.com/200x200/00FF00/FFFFFF?text=Prediction) |

---

## 🤖 Machine Learning Overview

### Machine Learning Workflow

![Machine Learning Workflow](https://miro.medium.com/v2/resize:fit:1200/1*oxy_4R--9EQfGyZusb1M5Q.png)

### Proses Machine Learning

Machine Learning adalah proses membuat komputer "belajar" dari data untuk membuat prediksi atau keputusan. Berikut adalah workflow umum dalam machine learning:

```
Data Collection → Data Preprocessing → Feature Engineering →
Model Training → Model Evaluation → Model Deployment
```

#### 1️⃣ **Data Collection**

- Mengumpulkan data (gambar medis, teks, angka, dll)
- Contoh: CT Scan, MRI, X-Ray untuk medical imaging

#### 2️⃣ **Data Preprocessing**

- Membersihkan data (remove noise, handle missing values)
- Normalisasi/standardisasi
- Data augmentation (untuk menambah variasi data)

#### 3️⃣ **Feature Engineering**

- Ekstraksi fitur penting dari data
- Untuk gambar: edges, textures, shapes
- Untuk text: word embeddings, TF-IDF

#### 4️⃣ **Model Training**

- Melatih model dengan data training
- Model belajar pattern dari data
- Optimasi hyperparameters

#### 5️⃣ **Model Evaluation**

- Test model dengan data yang belum pernah dilihat
- Metrics: Accuracy, Precision, Recall, F1-Score, dll
- Validasi performa model

#### 6️⃣ **Model Deployment**

- Deploy model ke production
- Monitoring performa
- Re-training jika perlu

---

### Tipe-Tipe Machine Learning

Machine Learning dibagi menjadi beberapa kategori berdasarkan tipe task-nya:

#### 🎯 **1. Supervised Learning**

Model belajar dari data yang sudah dilabeli (ada input dan output yang benar).

##### **A. Classification (Klasifikasi)**

**Tujuan**: Memprediksi kategori/kelas dari input

**Contoh Use Cases:**

- 🏥 **Medical**: Deteksi penyakit (normal vs abnormal)
- 📧 **Email**: Spam vs Not Spam
- 🖼️ **Image**: Cat vs Dog classification
- 💳 **Finance**: Fraud detection (fraud vs legitimate)

**Output**: Kategori diskrit (kelas A, B, C, dst)

```python
# Example
Input: Gambar X-Ray
Output: "Pneumonia" atau "Normal"
```

##### **B. Regression (Regresi)**

**Tujuan**: Memprediksi nilai numerik kontinu

**Contoh Use Cases:**

- 🏠 **Real Estate**: Prediksi harga rumah
- 📈 **Stock Market**: Prediksi harga saham
- 🌡️ **Weather**: Prediksi suhu besok
- 🏥 **Medical**: Prediksi umur pasien dari CT scan
- 💰 **Sales**: Prediksi revenue bulan depan

**Output**: Angka kontinu (0.5, 100.3, 1000, dst)

```python
# Example
Input: Luas tanah, jumlah kamar, lokasi
Output: Harga rumah = Rp 1,500,000,000
```

**Perbedaan Classification vs Regression:**

|            Aspek            |     Classification     |  Regression   |
| :-------------------------: | :--------------------: | :-----------: |
|         **Output**          |     Kategori/Label     | Angka kontinu |
|      **Contoh Output**      | "Cancer" atau "Benign" |  25.5, 100.3  |
|      **Loss Function**      |     Cross Entropy      |   MSE, MAE    |
| **Activation (last layer)** |    Softmax/Sigmoid     |    Linear     |
|   **Evaluation Metrics**    |   Accuracy, F1-Score   | MSE, RMSE, R² |

##### **C. Segmentation (Segmentasi)** ⭐ _Workshop Ini_

**Tujuan**: Memberi label pada setiap pixel/voxel dalam gambar

**Contoh Use Cases:**

- 🏥 **Medical**: Segmentasi organ (liver, kidney, tumor)
- 🚗 **Autonomous Driving**: Segmentasi jalan, mobil, pedestrian
- 🛰️ **Satellite**: Segmentasi lahan (hutan, kota, laut)
- 📸 **Photo Editing**: Background removal

**Output**: Mask/segmentation map (pixel-level labels)

```python
# Example - Workshop ini!
Input: CT Scan 3D (128x128x128)
Output: Segmentation mask (128x128x128)
        - 0 = background
        - 1 = organ target
```

**Tipe Segmentation:**

|           Tipe            |                        Penjelasan                        |        Use Case        |
| :-----------------------: | :------------------------------------------------------: | :--------------------: |
| **Semantic Segmentation** | Label per pixel, semua objek kelas sama punya label sama | Segmentasi jalan raya  |
| **Instance Segmentation** |        Label per pixel, objek yang sama dibedakan        | Deteksi multiple mobil |
| **Panoptic Segmentation** |               Gabungan semantic + instance               |   Autonomous driving   |

---

#### 🔍 **2. Unsupervised Learning**

Model belajar dari data tanpa label (hanya ada input, tidak ada output yang benar).

##### **A. Clustering (Pengelompokan)**

**Tujuan**: Mengelompokkan data yang mirip ke dalam cluster

**Contoh Use Cases:**

- 👥 **Customer Segmentation**: Kelompokkan customer berdasarkan perilaku
- 🏥 **Medical**: Kelompokkan pasien dengan gejala serupa
- 📰 **News**: Kelompokkan berita berdasarkan topik
- 🧬 **Genomics**: Kelompokkan gen dengan fungsi serupa
- 🛒 **E-commerce**: Product recommendation berdasarkan similarity

**Output**: Cluster ID (grup 0, 1, 2, dst)

```python
# Example
Input: Data pembelian customer (items, frequency, amount)
Output: Customer segmentation
        - Cluster 0: High spender
        - Cluster 1: Occasional buyer
        - Cluster 2: Window shopper
```

**Algoritma Clustering:**

- **K-Means**: Paling populer, fast, simple
- **DBSCAN**: Untuk data dengan noise
- **Hierarchical Clustering**: Membuat dendrogram
- **Gaussian Mixture Models**: Probabilistic clustering

##### **B. Dimensionality Reduction**

**Tujuan**: Mengurangi jumlah fitur sambil tetap mempertahankan informasi penting

**Contoh Use Cases:**

- 📊 **Visualization**: Plot data high-dimensional ke 2D/3D
- 🖼️ **Image Compression**: Compress gambar
- 🧬 **Genomics**: Reduce ribuan gen menjadi beberapa principal components
- 📈 **Feature Selection**: Pilih fitur terpenting

**Teknik:**

- **PCA** (Principal Component Analysis)
- **t-SNE**: Untuk visualisasi
- **UMAP**: Modern alternative to t-SNE
- **Autoencoders**: Deep learning approach

---

#### 🎮 **3. Reinforcement Learning**

Model belajar melalui trial-and-error dengan sistem reward/punishment.

**Contoh Use Cases:**

- 🎮 **Gaming**: AlphaGo, Chess AI
- 🤖 **Robotics**: Robot navigation
- 🚗 **Autonomous Driving**: Decision making
- 💰 **Trading**: Algorithmic trading
- 🎯 **Recommendation**: Personalized content

---

### Roadmap Belajar Machine Learning

Untuk workshop ini, kita fokus ke **Segmentation** (bagian dari Supervised Learning). Tapi Anda bisa eksplorasi tipe lain setelah mahir:

```
Level 1: Pemula
├── Classification (Image Classification)
├── Regression (Linear/Polynomial Regression)
└── Segmentation ⭐ (Workshop ini!)

Level 2: Intermediate
├── Object Detection (YOLO, Faster R-CNN)
├── Time Series (LSTM, Transformers)
└── Clustering (K-Means, DBSCAN)

Level 3: Advanced
├── GANs (Generative Adversarial Networks)
├── Reinforcement Learning
└── Transfer Learning & Fine-tuning
```

---

### Kenapa Kita Mulai dari Segmentation?

1. ✅ **Praktis**: Langsung aplikasi ke medical imaging
2. ✅ **Visual**: Hasil bisa langsung dilihat
3. ✅ **High Impact**: Medical AI sangat dibutuhkan
4. ✅ **Foundation**: Konsep bisa diterapkan ke task lain
5. ✅ **Job Market**: Medical AI engineer in high demand

**Setelah workshop ini**, Anda bisa adaptasi untuk:

- Classification: Ubah output layer & loss function
- Regression: Ganti output ke continuous value
- Clustering: Gunakan feature extraction dari encoder

---

### Comparison: Segmentation vs Other ML Tasks

|         Aspek          |  Classification   |   Regression   |    Segmentation    |       Clustering        |
| :--------------------: | :---------------: | :------------: | :----------------: | :---------------------: |
|    **Supervised?**     |      ✅ Yes       |     ✅ Yes     |       ✅ Yes       |          ❌ No          |
|    **Output Type**     |     Category      |     Number     |  Pixel-wise mask   |       Cluster ID        |
|  **Label Required?**   |      ✅ Yes       |     ✅ Yes     |       ✅ Yes       |          ❌ No          |
|    **Architecture**    |     CNN + FC      |  CNN + Linear  |     U-Net, FCN     |     K-Means, DBSCAN     |
|   **Loss Function**    |   Cross Entropy   |    MSE, MAE    |     Dice, IoU      | Within-cluster variance |
|     **Evaluation**     |   Accuracy, F1    |   RMSE, MAE    |     Dice, IoU      |    Silhouette score     |
| **Use Case (Medical)** | Disease detection | Age prediction | Organ segmentation |    Patient grouping     |

---

### 🎓 Ekstension untuk Task Lain

Setelah mahir segmentation, Anda bisa extend untuk task lain dengan sedikit modifikasi:

#### **Untuk Classification:**

```python
# Ubah model architecture
model = torch.nn.Sequential(
    UNet(...),  # Feature extractor
    torch.nn.AdaptiveAvgPool3d(1),  # Global pooling
    torch.nn.Flatten(),
    torch.nn.Linear(512, num_classes)  # Classification head
)

# Loss function
loss_fn = torch.nn.CrossEntropyLoss()
```

#### **Untuk Regression:**

```python
# Ubah output layer
model = torch.nn.Sequential(
    UNet(...),  # Feature extractor
    torch.nn.AdaptiveAvgPool3d(1),
    torch.nn.Flatten(),
    torch.nn.Linear(512, 1)  # Single output value
)

# Loss function
loss_fn = torch.nn.MSELoss()  # Mean Squared Error
```

#### **Untuk Clustering:**

```python
# Extract features menggunakan trained encoder
encoder = model.encoder  # From trained U-Net
features = encoder(images)

# Apply clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(features.cpu().numpy())
```

---

### 📚 Resources untuk Belajar Lebih Lanjut

**Classification:**

- [Image Classification with PyTorch](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

**Regression:**

- [Linear Regression Tutorial](https://scikit-learn.org/stable/modules/linear_model.html)
- [Deep Learning for Regression](https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/)

**Clustering:**

- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [K-Means Explained](https://www.youtube.com/watch?v=4b5d3muPQmA)

**General ML:**

- [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)

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

## 🎯 Try Pre-trained Model (Inference)

### ⭐ Repository Inference - Model Siap Pakai!

> **Tidak ingin training dari nol?** Langsung coba model AI yang sudah jadi!

**Repository**: [medical-ipynb - Learn Inference](https://github.com/scaidanpublikasi/medical-ipynb)

### 📦 Apa yang Ada di Repository Ini?

✅ **Model AI yang sudah ditraining** (`model.pt`)  
✅ **Sample gambar medis** untuk dicoba  
✅ **Script Python untuk inference** (prediksi otomatis)  
✅ **Dokumentasi lengkap** cara menggunakannya

### 🔄 Perbedaan dengan Repository Utama

| Repository                 |                   Fungsi                    |       Untuk       |
| :------------------------- | :-----------------------------------------: | :---------------: |
| `brats-segmentation`       |   **Training** - Cara membuat `model.pt`    | Belajar dari nol  |
| `medical-ipynb` ⭐ **NEW** | **Inference** - Cara menggunakan `model.pt` | Langsung praktek! |

### 🚀 Cara Menggunakan Model Pre-trained

#### 1️⃣ **Clone Repository Inference**

```bash
git clone https://github.com/scaidanpublikasi/medical-ipynb.git
cd medical-ipynb
```

#### 2️⃣ **Extract File (jika ada .zip)**

Repository ini berisi:

- `model.pt` → Model AI yang sudah ditraining
- `sample_images/` → Gambar medis untuk dicoba
- `inference.py` → Script Python untuk prediksi
- `requirements.txt` → Dependencies

```bash
# Jika ada file zip, extract dulu
unzip machine-learning-learn-inference.zip
```

#### 3️⃣ **Install Dependencies**

```bash
pip install -r requirements.txt
```

#### 4️⃣ **Run Inference Script**

```python
# Contoh penggunaan:
python inference.py --model_path model.pt --image_path sample_images/test_image.nii.gz
```

**Atau langsung di Python:**

```python
import torch
import nibabel as nib
import numpy as np
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ResizeWithPadOrCrop, NormalizeIntensity
from monai.networks.nets import UNet

# 1. Load model yang sudah jadi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=2
).to(device)

model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

# 2. Preprocessing gambar
transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ResizeWithPadOrCrop(spatial_size=(128, 128, 128)),
    NormalizeIntensity()
])

# 3. Load gambar sample
image = transforms("sample_images/test_image.nii.gz")
image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)

# 4. Prediksi!
with torch.no_grad():
    prediction = torch.sigmoid(model(image_tensor))
    pred_mask = (prediction > 0.5).cpu().numpy().squeeze()

# 5. Simpan hasil
output_img = nib.Nifti1Image(pred_mask.astype(np.uint8), affine=np.eye(4))
nib.save(output_img, "prediction_result.nii.gz")

print("✅ Prediksi selesai! Hasil tersimpan di: prediction_result.nii.gz")
```

#### 5️⃣ **Visualisasi Hasil**

```python
import matplotlib.pyplot as plt

# Ambil slice tengah untuk visualisasi
slice_idx = pred_mask.shape[2] // 2

plt.figure(figsize=(12, 4))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(image[0, :, :, slice_idx], cmap='gray')
plt.title('Original CT Scan')
plt.axis('off')

# Prediction
plt.subplot(1, 3, 2)
plt.imshow(pred_mask[:, :, slice_idx], cmap='jet')
plt.title('AI Prediction')
plt.axis('off')

# Overlay
plt.subplot(1, 3, 3)
plt.imshow(image[0, :, :, slice_idx], cmap='gray')
plt.imshow(pred_mask[:, :, slice_idx], cmap='jet', alpha=0.5)
plt.title('Overlay')
plt.axis('off')

plt.tight_layout()
plt.savefig('inference_result.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Visualisasi tersimpan di: inference_result.png")
```

### 📝 Catatan Penting

> 💡 **Tips untuk Peserta Workshop:**
>
> - Gunakan repository inference ini jika Anda ingin **langsung mencoba** model tanpa perlu training (hemat waktu!)
> - Gunakan repository utama (`brats-segmentation`) jika Anda ingin **belajar dari nol** cara membuat model

**Workflow Recommended:**

1. **Coba dulu** inference → Lihat hasilnya → Paham cara kerja model
2. **Baru belajar** training → Mengerti bagaimana model dibuat

### 🎓 Untuk Peserta Workshop

Anda bebas memilih jalur pembelajaran:

**🏃‍♂️ Jalur Cepat (Inference First):**

```
1. Clone medical-ipynb
2. Run inference script
3. Lihat hasil prediksi
4. Paham output model
→ Lalu belajar training di workshop utama
```

**📚 Jalur Lengkap (Training First):**

```
1. Ikuti workshop Part 1-2
2. Training model dari nol
3. Simpan model.pt
4. Gunakan untuk inference
```

### 📚 Resources Tambahan

- **GitHub Inference**: [https://github.com/scaidanpublikasi/medical-ipynb](https://github.com/scaidanpublikasi/medical-ipynb)
- **GitHub Training**: [Repository utama brats-segmentation]
- **Sample Data**: Tersedia di repository inference

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

# Visualization & metrics
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, jaccard_score
)
```

**Penjelasan:**

- **nibabel**: Membaca file medis format `.nii` atau `.nii.gz`
- **MONAI**: Framework khusus medical imaging (built on PyTorch)
- **torch**: Deep learning framework
- **sklearn**: Untuk menghitung metrics

---

### Section 2: Load Dataset

```python
# Path ke dataset
image_dir = "/kaggle/input/costa-adam/imagesTr"
label_dir = "/kaggle/input/costa-adam/labelsTr"
test_dir = "/kaggle/input/costa-adam/imagesTs"
label_test_dir = "/kaggle/input/costa-adam/labelsTs"

# List semua file gambar
image_paths = sorted([
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.endswith((".nii", ".nii.gz"))
])

label_paths = sorted([
    os.path.join(label_dir, f)
    for f in os.listdir(label_dir)
    if f.endswith((".nii", ".nii.gz"))
])

test_paths = sorted([
    os.path.join(test_dir, f)
    for f in os.listdir(test_dir)
    if f.endswith((".nii", ".nii.gz"))
])

label_test_paths = sorted([
    os.path.join(label_test_dir, f)
    for f in os.listdir(label_test_dir)
    if f.endswith((".nii", ".nii.gz"))
])

print(f"✅ Training images: {len(image_paths)}")
print(f"✅ Training labels: {len(label_paths)}")
print(f"✅ Test images: {len(test_paths)}")
print(f"✅ Test labels: {len(label_test_paths)}")
```

**Expected Output:**

```
✅ Training images: 100
✅ Training labels: 100
✅ Test images: 20
✅ Test labels: 20
```

---

### Section 3: Visualisasi Data

```python
# Ambil sample pertama
sample_image = nib.load(image_paths[0])
sample_label = nib.load(label_paths[0])

# Convert ke numpy array
image_data = sample_image.get_fdata()
label_data = sample_label.get_fdata()

print(f"Image shape: {image_data.shape}")
print(f"Label shape: {label_data.shape}")
print(f"Image min/max: {image_data.min():.2f} / {image_data.max():.2f}")
print(f"Label unique values: {np.unique(label_data)}")

# Visualisasi slice tengah
slice_idx = image_data.shape[2] // 2

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image_data[:, :, slice_idx], cmap='gray')
plt.title('CT Scan (Original)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(label_data[:, :, slice_idx], cmap='jet')
plt.title('Ground Truth (Label)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(image_data[:, :, slice_idx], cmap='gray')
plt.imshow(label_data[:, :, slice_idx], cmap='jet', alpha=0.5)
plt.title('Overlay')
plt.axis('off')

plt.tight_layout()
plt.show()
```

**Output**: Tiga gambar yang menunjukkan CT Scan asli, ground truth, dan overlay-nya.

---

### Section 4: Preprocessing & Augmentation

```python
# Define target size (untuk training lebih cepat)
input_size = (128, 128, 128)

# Transform pipeline
transforms = Compose([
    # 1. Load gambar (.nii.gz → tensor)
    LoadImaged(keys=["image", "label"]),

    # 2. Pastikan channel first (1, H, W, D)
    EnsureChannelFirstd(keys=["image", "label"]),

    # 3. Orientasi RAS (Right-Anterior-Superior)
    Orientationd(keys=["image", "label"], axcodes="RAS"),

    # 4. Resize ke ukuran tetap
    ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=input_size),

    # 5. Normalisasi intensitas (0-1)
    NormalizeIntensityd(keys="image"),

    # 6. Convert ke PyTorch tensor
    ToTensord(keys=["image", "label"])
])

print("✅ Transforms ready!")
```

**Penjelasan:**

- **LoadImaged**: Membaca file `.nii.gz`
- **EnsureChannelFirstd**: Format channel-first (seperti (C, H, W, D))
- **Orientationd**: Standarisasi orientasi gambar
- **ResizeWithPadOrCropd**: Resize ke ukuran tetap (128³)
- **NormalizeIntensityd**: Normalisasi pixel values
- **ToTensord**: Convert ke PyTorch tensor

---

### Section 5: Dataset Split

```python
# Buat dictionary untuk dataset
data_dicts = [
    {"image": img, "label": lbl}
    for img, lbl in zip(image_paths, label_paths)
]

# Split 80% training, 20% validation
train_len = int(0.8 * len(data_dicts))
val_len = len(data_dicts) - train_len

train_files, val_files = random_split(
    data_dicts,
    [train_len, val_len],
    generator=torch.Generator().manual_seed(42)  # Reproducibility
)

print(f"✅ Training samples: {train_len}")
print(f"✅ Validation samples: {val_len}")

# Create MONAI datasets
train_ds = MonaiDataset(train_files, transform=transforms)
val_ds = MonaiDataset(val_files, transform=transforms)

# Create data loaders
train_loader = DataLoader(
    train_ds,
    batch_size=1,  # 3D images are memory-heavy
    shuffle=True,
    num_workers=0  # Kaggle constraint
)

val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

print("✅ DataLoaders ready!")
```

**Output:**

```
✅ Training samples: 80
✅ Validation samples: 20
✅ DataLoaders ready!
```

---

### Section 6: Define Model

```python
# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

# Create U-Net model
model = UNet(
    spatial_dims=3,        # 3D data
    in_channels=1,         # Grayscale CT scan
    out_channels=1,        # Binary segmentation
    channels=(32, 64, 128, 256, 512),  # Feature maps per layer
    strides=(2, 2, 2, 2),  # Downsampling strides
    num_res_units=2        # Residual units per block
).to(device)

print(f"✅ Model created!")
print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**Output:**

```
🔧 Using device: cuda
✅ Model created!
📊 Total parameters: 15,234,561
```

**Penjelasan:**

- **spatial_dims=3**: Karena data 3D (x, y, z)
- **in_channels=1**: CT scan grayscale
- **out_channels=1**: Binary mask (organ vs background)
- **channels**: Jumlah filter di setiap level encoder/decoder
- **strides**: Faktor downsampling (2 = ukuran dibagi 2)

---

### Section 7: Loss Function & Optimizer

```python
# Dice-CE Loss (kombinasi Dice + Cross Entropy)
loss_fn = DiceCELoss(
    sigmoid=True,       # Aktivasi sigmoid
    squared_pred=False  # Untuk stabilitas
)

# Optimizer (Adam)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,           # Learning rate
    weight_decay=1e-5  # L2 regularization
)

print("✅ Loss function: Dice-CE Loss")
print("✅ Optimizer: Adam (lr=1e-4)")
```

**Penjelasan:**

- **DiceCELoss**: Gabungan Dice Score + Cross Entropy
  - **Dice**: Fokus pada overlap mask
  - **Cross Entropy**: Fokus pada pixel-wise classification
- **Adam**: Optimizer adaptif (paling populer untuk DL)

---

### Section 8: Training Loop

```python
# Training configuration
num_epochs = 1000
best_loss = float('inf')
patience = 5
trigger = 0

train_losses = []
val_losses = []

print("🚀 Starting training...\n")

for epoch in range(num_epochs):
    # ========== TRAINING ==========
    model.train()
    epoch_loss = 0

    for batch_idx, batch_data in enumerate(train_loader):
        # Move data to device
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Average training loss
    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)

    # ========== VALIDATION ==========
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for val_data in val_loader:
            val_inputs = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)
            val_outputs = model(val_inputs)
            val_loss += loss_fn(val_outputs, val_labels).item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    # Print progress
    print(f"Epoch {epoch+1:3d}/{num_epochs} | "
          f"Train Loss: {epoch_loss:.4f} | "
          f"Val Loss: {val_loss:.4f}")

    # ========== EARLY STOPPING ==========
    if val_loss < best_loss:
        best_loss = val_loss
        trigger = 0
        # Save best model
        torch.save(model.state_dict(), "/kaggle/working/best_model.pt")
        print(f"    ✅ New best model saved! (Val Loss: {best_loss:.4f})")
    else:
        trigger += 1
        print(f"    ⚠️ No improvement ({trigger}/{patience})")

        if trigger >= patience:
            print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
            break

print("\n🎉 Training completed!")
print(f"📊 Best validation loss: {best_loss:.4f}")
```

**Expected Output:**

```
🚀 Starting training...

Epoch   1/1000 | Train Loss: 0.6234 | Val Loss: 0.5891
    ✅ New best model saved! (Val Loss: 0.5891)
Epoch   2/1000 | Train Loss: 0.5123 | Val Loss: 0.4856
    ✅ New best model saved! (Val Loss: 0.4856)
...
Epoch  45/1000 | Train Loss: 0.1234 | Val Loss: 0.1356
    ⚠️ No improvement (1/5)
...
🛑 Early stopping triggered at epoch 50

🎉 Training completed!
📊 Best validation loss: 0.1289
```

---

### Section 9: Plot Training Progress

```python
plt.figure(figsize=(12, 6))

plt.plot(range(1, len(train_losses)+1), train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=4)
plt.plot(range(1, len(val_losses)+1), val_losses, 'r-x', label='Val Loss', linewidth=2, markersize=4)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Progress', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.savefig('/kaggle/working/training_curve.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Plot saved: training_curve.png")
```

---

## 📊 Evaluation Metrics

### Section 10: Evaluate on Validation Set

```python
# Load best model
model.load_state_dict(torch.load("/kaggle/working/best_model.pt"))
model.eval()

# Initialize metrics
metrics_list = {
    "Dice": [],
    "IoU": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1": []
}

print("🔍 Evaluating model...\n")

with torch.no_grad():
    for batch_data in val_loader:
        # Get predictions
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"]

        outputs = model(inputs)
        preds = torch.sigmoid(outputs).cpu().numpy()

        # Flatten arrays
        y_true = labels.numpy().flatten()
        y_pred = (preds > 0.5).astype(np.uint8).flatten()

        # Calculate metrics
        metrics_list["Dice"].append(f1_score(y_true, y_pred, zero_division=0))
        metrics_list["IoU"].append(jaccard_score(y_true, y_pred, zero_division=0))
        metrics_list["Accuracy"].append(accuracy_score(y_true, y_pred))
        metrics_list["Precision"].append(precision_score(y_true, y_pred, zero_division=0))
        metrics_list["Recall"].append(recall_score(y_true, y_pred, zero_division=0))
        metrics_list["F1"].append(f1_score(y_true, y_pred, zero_division=0))

# Print results
print("📊 VALIDATION METRICS")
print("=" * 40)
for metric, values in metrics_list.items():
    mean_val = np.mean(values)
    std_val = np.std(values)
    print(f"{metric:12s}: {mean_val:.4f} ± {std_val:.4f}")
```

**Expected Output:**

```
🔍 Evaluating model...

📊 VALIDATION METRICS
========================================
Dice        : 0.8567 ± 0.0234
IoU         : 0.7812 ± 0.0312
Accuracy    : 0.9456 ± 0.0123
Precision   : 0.8734 ± 0.0267
Recall      : 0.8423 ± 0.0289
F1          : 0.8567 ± 0.0234
```

---

### Section 11: Visualisasi Prediksi

```python
# Ambil sample dari validation set
sample_idx = 0
sample_batch = val_ds[sample_idx]

# Predict
with torch.no_grad():
    sample_input = sample_batch["image"].unsqueeze(0).to(device)
    sample_pred = torch.sigmoid(model(sample_input)).cpu().numpy()[0, 0]
    sample_pred_binary = (sample_pred > 0.5).astype(np.uint8)

# Get ground truth
sample_image = sample_batch["image"].cpu().numpy()[0]
sample_label = sample_batch["label"].cpu().numpy()[0]

# Visualize middle slice
slice_idx = sample_image.shape[2] // 2

plt.figure(figsize=(15, 5))

# Original
plt.subplot(1, 4, 1)
plt.imshow(sample_image[:, :, slice_idx], cmap='gray')
plt.title('Original CT Scan', fontsize=12, fontweight='bold')
plt.axis('off')

# Ground Truth
plt.subplot(1, 4, 2)
plt.imshow(sample_label[:, :, slice_idx], cmap='jet')
plt.title('Ground Truth', fontsize=12, fontweight='bold')
plt.axis('off')

# Prediction
plt.subplot(1, 4, 3)
plt.imshow(sample_pred_binary[:, :, slice_idx], cmap='jet')
plt.title('AI Prediction', fontsize=12, fontweight='bold')
plt.axis('off')

# Overlay
plt.subplot(1, 4, 4)
plt.imshow(sample_image[:, :, slice_idx], cmap='gray')
plt.imshow(sample_pred_binary[:, :, slice_idx], cmap='jet', alpha=0.5)
plt.title('Overlay', fontsize=12, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.savefig('/kaggle/working/prediction_sample.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Visualization saved: prediction_sample.png")
```

---

### Section 12: Evaluate on Test Set (Optional)

```python
# Prepare test data
test_dicts = [
    {"image": img, "label": lbl}
    for img, lbl in zip(test_paths, label_test_paths)
]

test_ds = MonaiDataset(test_dicts, transform=transforms)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

# Evaluate
test_metrics = {
    "Dice": [],
    "IoU": []
}

print("🧪 Testing on unseen data...\n")

with torch.no_grad():
    for test_data in test_loader:
        inputs = test_data["image"].to(device)
        labels = test_data["label"]

        outputs = model(inputs)
        preds = torch.sigmoid(outputs).cpu().numpy()

        y_true = labels.numpy().flatten()
        y_pred = (preds > 0.5).astype(np.uint8).flatten()

        test_metrics["Dice"].append(f1_score(y_true, y_pred, zero_division=0))
        test_metrics["IoU"].append(jaccard_score(y_true, y_pred, zero_division=0))

print("📊 TEST SET METRICS")
print("=" * 40)
print(f"Dice Score: {np.mean(test_metrics['Dice']):.4f}")
print(f"IoU Score : {np.mean(test_metrics['IoU']):.4f}")
```

---

## 🐛 Troubleshooting

### Common Issues & Solutions

#### 1. Out of Memory (OOM) Error

**Error:**

```
RuntimeError: CUDA out of memory
```

**Solutions:**

```python
# A. Reduce batch size
train_loader = DataLoader(train_ds, batch_size=1)  # Default: 1

# B. Reduce input size
input_size = (96, 96, 96)  # Instead of (128, 128, 128)

# C. Use gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    loss = loss_fn(outputs, labels) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# D. Enable mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

#### 2. Kaggle GPU Not Available

**Error:**

```
Using device: cpu
```

**Solution:**

1. Click **"Settings"** (⚙️) on right panel
2. Select **"Accelerator" → "GPU T4"**
3. Click **"Save"**
4. Restart notebook: **"Session" → "Restart Session"**

---

#### 3. Dataset Not Found

**Error:**

```
FileNotFoundError: [Errno 2] No such file or directory
```

**Solution:**

```python
# Check available datasets
import os
print(os.listdir('/kaggle/input'))

# Add dataset manually:
# 1. Click "+ Add Data" (right panel)
# 2. Search: "costa adam"
# 3. Click "Add"
```

---

#### 4. Module Not Found

**Error:**

```
ModuleNotFoundError: No module named 'monai'
```

**Solution:**

```python
# Install missing packages
!pip install monai nibabel -q

# Restart kernel: Kernel → Restart
```

---

#### 5. Training Too Slow

**Solutions:**

```python
# A. Use smaller input size
input_size = (64, 64, 64)  # Faster but less accurate

# B. Reduce model complexity
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128),  # Fewer channels
    strides=(2, 2, 2),
    num_res_units=1
)

# C. Train fewer epochs
num_epochs = 50  # Instead of 1000

# D. Use smaller dataset
train_files = train_files[:20]  # First 20 samples only
```

---

#### 6. Dice Score Too Low (<0.5)

**Possible Causes & Fixes:**

```python
# A. Check class imbalance
print(f"Positive pixels: {labels.sum() / labels.numel() * 100:.2f}%")

# If <5%: Use weighted loss
loss_fn = DiceCELoss(sigmoid=True, lambda_dice=0.7, lambda_ce=0.3)

# B. Adjust learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)  # Smaller LR

# C. Add data augmentation
from monai.transforms import RandFlipd, RandRotate90d

transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
    # ... (other transforms)
])
```

---

#### 7. Notebook Crashes

**Solutions:**

1. **Save frequently**: `Ctrl + S`
2. **Enable auto-save**: Settings → Auto-save every 2 minutes
3. **Reduce memory usage**: Close unused tabs
4. **Restart session**: Session → Restart Session

---

## ❓ FAQ

### General Questions

**Q1: Berapa lama training-nya?**

A: Tergantung GPU dan dataset size:

- GPU T4 (Kaggle): ~2-3 jam (100 epochs)
- GPU P100 (Kaggle): ~1-2 jam
- CPU: ❌ Tidak disarankan (sangat lambat!)

---

**Q2: Apakah bisa run di CPU?**

A: Bisa, tapi **sangat lambat** (10-100x lebih lambat dari GPU).

```python
# Cek device
device = torch.device("cpu")  # Force CPU
```

---

**Q3: Dataset apa saja yang bisa dipakai?**

A: Format medical imaging yang didukung:

- ✅ `.nii` / `.nii.gz` (NIfTI)
- ✅ `.dcm` (DICOM)
- ✅ `.mha` / `.mhd` (MetaImage)
- ✅ `.nrrd` (NRRD)

---

**Q4: Apakah bisa untuk segmentasi multi-class?**

A: Ya! Ubah `out_channels`:

```python
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,  # 3 classes (background, organ1, organ2)
    # ...
)

# Use softmax instead of sigmoid
loss_fn = DiceCELoss(softmax=True)
```

---

**Q5: Bagaimana cara deploy model?**

A:

```python
# 1. Save model
torch.save(model.state_dict(), "model.pt")

# 2. Load di environment baru
model = UNet(...)
model.load_state_dict(torch.load("model.pt"))
model.eval()

# 3. Inference
with torch.no_grad():
    prediction = model(new_image)
```

---

### Technical Questions

**Q6: Apa bedanya Dice Loss dan Cross Entropy?**

A:

|     Metric     |            Focus            |      Best For       |
| :------------: | :-------------------------: | :-----------------: |
|   Dice Loss    |   Region overlap (global)   | Imbalanced datasets |
| Cross Entropy  | Pixel-wise accuracy (local) |  Balanced datasets  |
| **Dice-CE** ✅ |      Gabungan keduanya      |  **Recommended!**   |

---

**Q7: Kenapa perlu Early Stopping?**

A:

- Mencegah **overfitting** (model terlalu "hafal" training data)
- **Hemat waktu** (stop jika tidak ada improvement)
- **Pilih model terbaik** (berdasarkan validation loss)

---

**Q8: Apa itu Skip Connections di U-Net?**

A:

```
Encoder                Decoder
  [64] ───────────────→ [64]
    ↓                     ↑
  [128] ──────────────→ [128]
    ↓                     ↑
  [256] ──────────────→ [256]
```

**Fungsi:**

- Gabungkan **detail halus** (dari encoder) dengan **semantik kasar** (dari decoder)
- Hasil: Segmentation lebih akurat!

---

**Q9: Apa itu Sliding Window Inference?**

A: Teknik prediksi untuk gambar besar:

```python
from monai.inferers import sliding_window_inference

# Predict dengan window kecil
prediction = sliding_window_inference(
    inputs=large_image,
    roi_size=(96, 96, 96),  # Window size
    sw_batch_size=4,
    predictor=model,
    overlap=0.5
)
```

**Keuntungan:**

- ✅ Bisa handle gambar sangat besar
- ✅ Lebih stabil (rata-rata prediksi overlapping windows)

---

**Q10: Bagaimana cara improve model?**

A:

1. **Data augmentation**:

   ```python
   from monai.transforms import RandFlipd, RandRotate90d, RandGaussianNoised

   transforms = Compose([
       # ... (transforms lain)
       RandFlipd(keys=["image", "label"], prob=0.5),
       RandRotate90d(keys=["image", "label"], prob=0.3),
       RandGaussianNoised(keys="image", prob=0.2),
   ])
   ```

2. **Deeper model**:

   ```python
   model = UNet(
       channels=(32, 64, 128, 256, 512, 1024),  # More layers
       strides=(2, 2, 2, 2, 2)
   )
   ```

3. **Learning rate scheduling**:

   ```python
   from torch.optim.lr_scheduler import ReduceLROnPlateau

   scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)
   scheduler.step(val_loss)
   ```

4. **Ensemble models**:
   ```python
   # Train 3 models, average predictions
   pred_final = (pred1 + pred2 + pred3) / 3
   ```

---

## 📚 Resources

### Official Documentation

- [MONAI Docs](https://docs.monai.io/) - Medical imaging toolkit
- [PyTorch Docs](https://pytorch.org/docs/) - Deep learning framework
- [Kaggle Learn](https://www.kaggle.com/learn) - Free courses
- [NiBabel](https://nipy.org/nibabel/) - Medical image I/O

### Papers & Articles

- [U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)](https://arxiv.org/abs/1505.04597)
- [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation (2016)](https://arxiv.org/abs/1606.06650)
- [nnU-Net: Self-configuring Method for Deep Learning-based Biomedical Image Segmentation (2018)](https://arxiv.org/abs/1809.10486)

### Datasets

- [Medical Segmentation Decathlon](http://medicaldecathlon.com/) - 10 medical segmentation tasks
- [BraTS Challenge](https://www.med.upenn.edu/cbica/brats/) - Brain tumor segmentation
- [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) - Lung CT scans

### Video Tutorials

- [U-Net Explained (YouTube)](https://www.youtube.com/watch?v=azM57JuQpQI)
- [Medical Image Analysis with Deep Learning](https://www.coursera.org/learn/medical-image-analysis)

### Community

- [Kaggle Medical Imaging Forum](https://www.kaggle.com/discussions)
- [MONAI GitHub](https://github.com/Project-MONAI/MONAI)
- [Reddit r/computervision](https://www.reddit.com/r/computervision/)

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Ways to Contribute

1. **Report bugs**: Open an issue on GitHub
2. **Suggest features**: Submit feature requests
3. **Improve documentation**: Fix typos, add examples
4. **Add code**: Submit pull requests

### Contribution Guidelines

```bash
# 1. Fork the repository
# 2. Create a branch
git checkout -b feature/your-feature-name

# 3. Make changes
# 4. Commit with clear message
git commit -m "Add: Feature description"

# 5. Push to your fork
git push origin feature/your-feature-name

# 6. Open a Pull Request
```

### Code Style

- Follow PEP 8 (Python style guide)
- Add docstrings to functions
- Include type hints
- Write unit tests for new features

---

## 📜 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 SCAIDAN Publikasi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[Full license text...]
```

**TL;DR**: You can use this code freely for any purpose (commercial/non-commercial), but must include the license notice.

---

## 🙏 Acknowledgments

Special thanks to:

- **MONAI Team** - For the amazing medical imaging toolkit
- **Kaggle** - For free GPU compute
- **Contributors** - Everyone who helped improve this workshop
- **Medical AI Community** - For inspiring this project

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/scaidanpublikasi/medical-ipynb?style=social)
![GitHub forks](https://img.shields.io/github/forks/scaidanpublikasi/medical-ipynb?style=social)
![GitHub issues](https://img.shields.io/github/issues/scaidanpublikasi/medical-ipynb)
![GitHub pull requests](https://img.shields.io/github/issues-pr/scaidanpublikasi/medical-ipynb)

---

## 📈 Roadmap

### Current (v1.0.0)

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
- ✅ Using pre-trained models for inference ⭐

**Remember:**

> "The expert in anything was once a beginner." - Helen Hayes

Keep learning, keep coding, and don't be afraid to experiment!

**Next Steps:**

1. ⭐ Star this repository
2. 🍴 Fork and customize for your projects
3. 📢 Share with others learning AI
4. 💬 Join our community
5. 🚀 Start your own medical AI project!
6. 🎯 Try the inference repository untuk praktek langsung!

---

**Made with ❤️ for AI learners worldwide**

_Last updated: October 24, 2025_

---

**[⬆ Back to Top](#-workshop-medical-image-segmentation-dengan-u-net)**
