# 🚀 Setup Jupyter Notebook & Google Colab untuk Pemula

[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

> **Panduan Lengkap Setup Environment Machine Learning**  
> Level: Pemula Banget (Gak Perlu Pengalaman Coding!)

---

## 📋 Daftar Isi

- [Apa Itu Jupyter & Colab?](#-apa-itu-jupyter--colab)
- [Perbandingan Platform](#-perbandingan-platform-kaggle-vs-colab-vs-jupyter)
- [Setup Google Colab](#-setup-google-colab-paling-mudah)
- [Setup Jupyter Notebook (Lokal)](#-setup-jupyter-notebook-lokal)
- [Tips & Trik](#-tips--trik)
- [Troubleshooting](#-troubleshooting)
- [FAQ](#-faq)

---

## 🤔 Apa Itu Jupyter & Colab?

### Analogi Sederhana:

```
Jupyter Notebook = Microsoft Word untuk Coding
Google Colab     = Google Docs untuk Coding (online!)
Kaggle           = Google Docs + Komputer Super Gratis
```

### Penjelasan:

**Jupyter Notebook:**

- 📝 Aplikasi untuk menulis & menjalankan kode Python
- 💻 Berjalan di komputer sendiri (offline)
- 🎨 Bisa customisasi sesuka hati
- 🔒 Data & kode tersimpan di komputer Anda

**Google Colab:**

- ☁️ Jupyter Notebook versi online (di browser)
- 🆓 Gratis dengan GPU/TPU dari Google
- 📱 Bisa diakses dari mana saja
- 💾 Tersimpan di Google Drive

**Kaggle:**

- 🏆 Platform kompetisi data science
- 💪 GPU gratis lebih kuat (30 jam/minggu)
- 📊 Akses dataset gratis ribuan
- 👥 Komunitas besar

---

## 📊 Perbandingan Platform: Kaggle vs Colab vs Jupyter

### Tabel Perbandingan Lengkap:

| Fitur                | Kaggle ⭐⭐⭐⭐⭐          | Google Colab ⭐⭐⭐⭐       | Jupyter Local ⭐⭐⭐ |
| -------------------- | -------------------------- | --------------------------- | -------------------- |
| **Harga**            | 🆓 Gratis                  | 🆓 Gratis (ada premium)     | 🆓 Gratis            |
| **GPU Gratis**       | ✅ P100/T4 (30 jam/minggu) | ✅ T4 (terbatas)            | ❌ Tergantung PC     |
| **TPU Gratis**       | ❌ Tidak ada               | ✅ Ada                      | ❌ Tidak ada         |
| **RAM**              | 13-30 GB                   | 12 GB (25 GB di Pro)        | Tergantung PC        |
| **Storage**          | 20 GB                      | 15 GB (Google Drive)        | Tergantung PC        |
| **Internet**         | ✅ Perlu                   | ✅ Perlu                    | ❌ Tidak perlu       |
| **Setup**            | Sangat mudah               | Sangat mudah                | Butuh install        |
| **Dataset**          | ✅ Ribuan dataset built-in | ❌ Harus upload             | ❌ Harus download    |
| **Komunitas**        | ✅ Besar                   | ⚠️ Sedang                   | ❌ Tidak ada         |
| **Privacy**          | ⚠️ Public/Private          | ⚠️ Google punya akses       | ✅ Full control      |
| **Kolaborasi**       | ✅ Mudah share             | ✅ Real-time collab         | ❌ Susah             |
| **Kecepatan Upload** | ⚡ Cepat (server-side)     | 🐌 Lambat (upload ke Drive) | ⚡ Lokal             |
| **Stabilitas**       | ⭐⭐⭐⭐⭐                 | ⭐⭐⭐⭐                    | ⭐⭐⭐⭐⭐           |

---

### 🎯 Kapan Pakai Yang Mana?

#### Pakai **KAGGLE** kalau:

✅ Belajar machine learning dari nol  
✅ Butuh GPU gratis yang kuat  
✅ Mau akses dataset gratis  
✅ Ikut kompetisi data science  
✅ Belajar dari komunitas

**Contoh Use Case:**

- Workshop medical imaging (seperti materi kita!)
- Kompetisi ML
- Training model deep learning

---

#### Pakai **GOOGLE COLAB** kalau:

✅ Mau kolaborasi real-time dengan tim  
✅ Butuh akses dari berbagai device  
✅ Dataset kecil (< 5 GB)  
✅ Prototyping cepat  
✅ Sudah familiar dengan Google Drive

**Contoh Use Case:**

- Kerja kelompok
- Demo & presentasi
- Eksperimen cepat
- Tutorial online

---

#### Pakai **JUPYTER LOCAL** kalau:

✅ Data sensitif/private  
✅ Dataset sangat besar (> 100 GB)  
✅ Punya GPU sendiri  
✅ Internet tidak stabil  
✅ Butuh customisasi penuh

**Contoh Use Case:**

- Research lab
- Production environment
- Data perusahaan
- Offline development

---

### 💡 Rekomendasi untuk Pemula:

```
1. Mulai dengan KAGGLE (paling mudah + GPU gratis kuat)
   ↓
2. Coba GOOGLE COLAB (belajar kolaborasi)
   ↓
3. Install JUPYTER LOCAL (kalau sudah advanced)
```

---

## ☁️ Setup Google Colab (Paling Mudah!)

### ⏱️ Waktu Setup: 2 Menit

### Step 1: Buka Google Colab

1. Buka browser (Chrome recommended)
2. Ketik: `colab.research.google.com`
3. Login dengan akun Google Anda
4. **Done!** 🎉

### Step 2: Buat Notebook Baru

**Cara 1: Dari Homepage**

```
File → New Notebook
```

**Cara 2: Langsung**

- Klik tombol **"New Notebook"** (pojok kiri bawah)

**Cara 3: Dari Google Drive**

```
Google Drive → New → More → Google Colaboratory
```

### Step 3: Rename Notebook

```python
# Klik "Untitled0.ipynb" di pojok kiri atas
# Ganti nama, misal: "Medical_AI_Project.ipynb"
```

### Step 4: Enable GPU (PENTING!)

```
Runtime → Change runtime type → Hardware accelerator → GPU → Save
```

**Verifikasi GPU aktif:**

```python
# Run cell ini untuk cek GPU
import torch
print("GPU tersedia:", torch.cuda.is_available())
print("Nama GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Tidak ada GPU")

# Cek spesifikasi
!nvidia-smi
```

**Output yang diharapkan:**

```
GPU tersedia: True
Nama GPU: Tesla T4

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   XX°C    P0    XX W / XX W |      0MiB / 15360MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

---

### Step 5: Upload Dataset ke Google Drive

**Cara 1: Manual Upload**

1. Buka Google Drive: `drive.google.com`
2. Buat folder baru: "AI_Datasets"
3. Upload file dataset Anda
4. Copy path folder

**Cara 2: Mount Google Drive di Colab**

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Sekarang bisa akses Drive
import os
print(os.listdir('/content/drive/MyDrive'))
```

**Contoh akses dataset:**

```python
# Misal dataset ada di: MyDrive/AI_Datasets/medical_images/

data_path = '/content/drive/MyDrive/AI_Datasets/medical_images'

# List file
import os
files = os.listdir(data_path)
print(f"Total files: {len(files)}")
print(f"5 file pertama: {files[:5]}")
```

---

### Step 6: Install Library

```python
# Install library yang dibutuhkan
!pip install monai nibabel

# Atau install dari requirements.txt
!pip install -r /content/drive/MyDrive/requirements.txt
```

---

### Step 7: Save Notebook

**Auto-save:**

- Colab otomatis save setiap beberapa menit ke Google Drive

**Manual save:**

```
File → Save
atau
Ctrl + S (Windows/Linux)
Cmd + S (Mac)
```

**Download notebook:**

```
File → Download → Download .ipynb
```

---

### 🎁 Bonus: Upload Dataset Langsung ke Colab

**Untuk dataset kecil (< 100 MB):**

```python
from google.colab import files

# Upload file
uploaded = files.upload()

# Lihat file yang terupload
import os
print(os.listdir())
```

**Untuk dataset besar, gunakan Google Drive atau link download:**

```python
# Download dari URL
!wget https://example.com/dataset.zip

# Extract
!unzip dataset.zip

# Atau pakai gdown untuk Google Drive link
!pip install gdown
!gdown --id YOUR_FILE_ID
```

---

### ⚡ Tips Google Colab:

**1. Hindari Disconnect:**

```javascript
// Paste di Console browser (F12 → Console)
// Untuk mencegah disconnect saat idle
function ClickConnect() {
  console.log("Clicking connect button");
  document.querySelector("colab-toolbar-button#connect").click();
}
setInterval(ClickConnect, 60000);
```

**2. Monitor Resource:**

```python
# Cek RAM & Disk usage
!free -h
!df -h
```

**3. Clear Output (hemat RAM):**

```
Edit → Clear all outputs
```

**4. Keyboard Shortcuts:**
| Shortcut | Fungsi |
|----------|--------|
| `Ctrl + M + B` | Insert cell below |
| `Ctrl + M + A` | Insert cell above |
| `Ctrl + M + D` | Delete cell |
| `Ctrl + /` | Comment code |
| `Shift + Enter` | Run cell |

---

## 💻 Setup Jupyter Notebook (Lokal)

### ⏱️ Waktu Setup: 15-30 Menit

### Prerequisites:

- 💾 Minimal 5 GB free space
- 🌐 Internet untuk download
- 🖥️ Windows/Mac/Linux

---

### Metode 1: Install via Anaconda (Recommended untuk Pemula)

#### Step 1: Download Anaconda

1. Buka: `https://www.anaconda.com/download`
2. Pilih sesuai OS Anda:
   - Windows: Download `.exe`
   - Mac: Download `.pkg`
   - Linux: Download `.sh`

**Ukuran file:** ~500 MB

#### Step 2: Install Anaconda

**Windows:**

```
1. Double-click file .exe
2. Click "Next" → "I Agree"
3. Install untuk "Just Me" (recommended)
4. Pilih folder instalasi (default: C:\Users\YourName\anaconda3)
5. ✅ Centang "Add Anaconda to PATH" (PENTING!)
6. Click "Install" → tunggu selesai (~10 menit)
7. Click "Finish"
```

**Mac:**

```
1. Double-click file .pkg
2. Follow installer instructions
3. Agree to license
4. Install → masukkan password Mac
5. Close installer
```

**Linux:**

```bash
# Buka Terminal
cd ~/Downloads
bash Anaconda3-xxxx-Linux-x86_64.sh

# Follow prompts:
# - Enter untuk baca license
# - Ketik "yes" untuk agree
# - Enter untuk default location
# - Ketik "yes" untuk conda init
```

#### Step 3: Verifikasi Instalasi

**Windows:**

```
1. Buka "Anaconda Prompt" (cari di Start Menu)
2. Ketik: conda --version
```

**Mac/Linux:**

```bash
# Buka Terminal
conda --version
```

**Expected output:**

```
conda 23.x.x
```

#### Step 4: Buat Environment Baru

```bash
# Buat environment untuk medical AI
conda create -n medical-ai python=3.10

# Aktivasi environment
# Windows:
conda activate medical-ai

# Mac/Linux:
source activate medical-ai
```

**Kenapa pakai environment?**

- ✅ Isolasi project
- ✅ Hindari konflik library
- ✅ Mudah management
- ✅ Bisa punya banyak project dengan versi library berbeda

#### Step 5: Install Jupyter

```bash
# Install Jupyter Notebook
conda install jupyter

# Atau pakai pip
pip install jupyter
```

#### Step 6: Install Library ML

```bash
# Install library untuk medical imaging
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install MONAI & dependencies
pip install monai nibabel numpy matplotlib scikit-learn pandas

# Atau install semuanya sekaligus
pip install torch torchvision monai nibabel numpy matplotlib scikit-learn pandas
```

#### Step 7: Jalankan Jupyter

```bash
# Start Jupyter Notebook
jupyter notebook

# Atau dengan port spesifik
jupyter notebook --port=8888
```

**Browser akan otomatis terbuka dengan URL:**

```
http://localhost:8888/tree
```

#### Step 8: Buat Notebook Pertama

1. Di browser, klik **"New"** → **"Python 3"**
2. Rename notebook: klik "Untitled" → ganti nama
3. Mulai coding!

**Test installation:**

```python
# Cell 1: Test import
import torch
import numpy as np
import matplotlib.pyplot as plt

print("✅ PyTorch version:", torch.__version__)
print("✅ NumPy version:", np.__version__)

# Cell 2: Test GPU (kalau punya NVIDIA GPU)
if torch.cuda.is_available():
    print("🎉 GPU tersedia!")
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️ GPU tidak tersedia, pakai CPU")
```

---

### Metode 2: Install via pip (Advanced)

**Untuk yang sudah punya Python installed:**

```bash
# Install Python dari python.org (versi 3.8-3.11)

# Install Jupyter via pip
pip install jupyter

# Install ML libraries
pip install torch torchvision monai nibabel numpy matplotlib scikit-learn

# Run Jupyter
jupyter notebook
```

---

### 🎨 Customize Jupyter

**1. Install Jupyter Extensions:**

```bash
# Install nbextensions
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# Enable extensions
jupyter nbextension enable codefolding/main
jupyter nbextension enable execute_time/ExecuteTime
```

**2. Install JupyterLab (Jupyter versi baru):**

```bash
pip install jupyterlab

# Run JupyterLab
jupyter lab
```

**3. Theme gelap:**

```bash
# Install jupyterthemes
pip install jupyterthemes

# Aktifkan theme
jt -t onedork -fs 12 -nfs 13 -tfs 13 -ofs 11 -cellw 95%

# Reset ke default
jt -r
```

**4. Add kernel environment:**

```bash
# Install ipykernel
conda install ipykernel

# Add environment ke Jupyter
python -m ipykernel install --user --name=medical-ai --display-name="Medical AI"
```

---

## 📂 Struktur Folder Recommended

```
Projects/
├── Medical_AI_Workshop/
│   ├── notebooks/
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 02_preprocessing.ipynb
│   │   ├── 03_model_training.ipynb
│   │   └── 04_evaluation.ipynb
│   ├── data/
│   │   ├── raw/              # Dataset asli
│   │   ├── processed/        # Dataset sudah diproses
│   │   └── predictions/      # Hasil prediksi
│   ├── models/
│   │   ├── checkpoints/      # Model checkpoints
│   │   └── best_model.pt     # Model terbaik
│   ├── src/                  # Source code Python
│   │   ├── preprocessing.py
│   │   ├── model.py
│   │   └── utils.py
│   ├── results/
│   │   ├── figures/          # Grafik & visualisasi
│   │   └── metrics/          # Hasil evaluasi
│   ├── requirements.txt      # List library
│   └── README.md            # Dokumentasi project
```

---

## 🔄 Migrasi dari Kaggle ke Colab/Jupyter

### Step-by-Step:

**1. Download Notebook dari Kaggle:**

```
Di Kaggle: File → Download Notebook
Simpan file .ipynb
```

**2. Upload ke Colab:**

```
Di Colab: File → Upload notebook
Pilih file .ipynb yang di-download
```

**3. Upload ke Jupyter Local:**

```
Di Jupyter: Upload button (pojok kanan atas)
Pilih file .ipynb
```

**4. Sesuaikan Path Dataset:**

**Kaggle:**

```python
data_path = "/kaggle/input/dataset-name/"
```

**Google Colab:**

```python
from google.colab import drive
drive.mount('/content/drive')
data_path = "/content/drive/MyDrive/dataset-name/"
```

**Jupyter Local:**

```python
data_path = "C:/Users/YourName/Projects/data/dataset-name/"
# atau untuk Mac/Linux:
data_path = "/Users/YourName/Projects/data/dataset-name/"
```

**5. Install Library yang Kurang:**

```python
# Di cell pertama notebook
!pip install monai nibabel

# Import libraries
import monai
import nibabel as nib
```

---

## 💡 Tips & Trik

### 🚀 Productivity Tips:

**1. Magic Commands (Jupyter):**

```python
# Lihat semua magic commands
%lsmagic

# Timer eksekusi
%time code_here
%timeit code_here

# Lihat variabel
%whos

# Clear output
%clear

# Run external Python file
%run script.py

# Matplotlib inline
%matplotlib inline
```

**2. Debugging:**

```python
# Interactive debugger
%debug

# Jalankan cell dalam debug mode
%%debug
```

**3. Shortcuts:**

| Shortcut        | Fungsi              |
| --------------- | ------------------- |
| `Esc + A`       | Insert cell above   |
| `Esc + B`       | Insert cell below   |
| `Esc + D + D`   | Delete cell         |
| `Esc + M`       | Convert to Markdown |
| `Esc + Y`       | Convert to Code     |
| `Shift + Enter` | Run & next cell     |
| `Ctrl + Enter`  | Run current cell    |
| `Alt + Enter`   | Run & insert below  |

**4. Best Practices:**

```python
# ✅ Good: Import di awal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ✅ Good: Clear variable names
training_loss = 0.5234

# ❌ Bad: Single letter (kecuali loop)
t = 0.5234

# ✅ Good: Add comments
# Calculate mean Dice score
mean_dice = np.mean(dice_scores)

# ✅ Good: Section headers
# ============================================
# DATA PREPROCESSING
# ============================================
```

---

### 🔧 Performance Tips:

**1. Monitor Memory:**

```python
# Cek memory usage
import psutil
import os

def check_memory():
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1024 ** 3
    print(f"Memory usage: {memory_gb:.2f} GB")

check_memory()
```

**2. Clear Memory:**

```python
# Delete variables
del large_variable

# Clear all variables
%reset -f

# Garbage collection
import gc
gc.collect()
```

**3. Parallel Processing:**

```python
# Use multiprocessing
from multiprocessing import Pool

def process_image(image_path):
    # Process logic
    return result

# Parallel execution
with Pool(processes=4) as pool:
    results = pool.map(process_image, image_paths)
```

---

## 🐛 Troubleshooting

### Problem 1: "Kernel Dead" / "Kernel Crashed"

**Symptoms:**

```
The kernel appears to have died. It will restart automatically.
```

**Causes & Solutions:**

```python
# Cause 1: Out of Memory
# Solution: Reduce batch size, clear variables

# Check memory before crash
import psutil
mem = psutil.virtual_memory()
print(f"Available RAM: {mem.available / 1024**3:.2f} GB")

# Cause 2: Infinite loop
# Solution: Add break condition atau timeout

# Cause 3: Incompatible library
# Solution: Update libraries
!pip install --upgrade torch torchvision
```

---

### Problem 2: "Module Not Found"

**Symptoms:**

```
ModuleNotFoundError: No module named 'monai'
```

**Solutions:**

```python
# Solution 1: Install module
!pip install monai

# Solution 2: Check if correct environment
!which python  # Should point to your conda env

# Solution 3: Restart kernel after install
# Kernel → Restart Kernel
```

---

### Problem 3: Google Colab Disconnected

**Symptoms:**

- Browser tab disconnected
- "Reconnecting..." message

**Solutions:**

```javascript
// Paste in browser console (F12)
function ClickConnect() {
  console.log("Keep alive");
  document.querySelector("colab-toolbar-button#connect").click();
}
setInterval(ClickConnect, 60000);
```

**Or use:**

```python
# Install colab-alive
!pip install colab-alive

# Keep session alive
from colab_alive import keep_alive
keep_alive()
```

---

### Problem 4: "GPU Out of Memory"

**Symptoms:**

```
RuntimeError: CUDA out of memory
```

**Solutions:**

```python
# Solution 1: Clear cache
import torch
torch.cuda.empty_cache()

# Solution 2: Reduce batch size
batch_size = 1  # Smallest possible

# Solution 3: Use gradient accumulation
accumulation_steps = 4

# Solution 4: Use smaller model
model = UNet(channels=(16, 32, 64, 128))  # Smaller channels

# Solution 5: Mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

---

### Problem 5: Slow Upload to Google Drive

**Solutions:**

```python
# Method 1: Use gdown for large files
!pip install gdown
!gdown --id FILE_ID

# Method 2: Download from URL
!wget https://example.com/dataset.zip

# Method 3: Use rsync (for Drive sync)
# Mount Drive → copy using rsync instead of cp
```

---

## ❓ FAQ

### Q1: Kaggle vs Colab vs Jupyter, mana yang paling bagus?

**A:** Tergantung kebutuhan:

- **Pemula + Butuh GPU kuat** → Kaggle ⭐⭐⭐⭐⭐
- **Kolaborasi tim** → Google Colab ⭐⭐⭐⭐
- **Data sensitive + Offline** → Jupyter Local ⭐⭐⭐⭐

---

### Q2: Apakah bisa pakai GPU di Jupyter lokal?

**A:** Bisa, kalau PC/laptop punya NVIDIA GPU!

**Check GPU:**

```python
import torch
print(torch.cuda.is_available())
```

**Install CUDA:**

1. Download CUDA Toolkit: `developer.nvidia.com/cuda-downloads`
2. Install sesuai OS
3. Install PyTorch dengan CUDA:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### Q3: Google Colab gratis atau bayar?

**A:** Ada 2 versi:

| Fitur           | Free          | Pro ($9.99/month) |
| --------------- | ------------- | ----------------- |
| GPU             | T4 (terbatas) | V100/A100         |
| RAM             | 12 GB         | 25 GB             |
| Runtime         | 12 jam max    | 24 jam max        |
| Background exec | ❌            | ✅                |

**Rekomendasi:** Mulai dari gratis dulu!

---

### Q4: Bagaimana cara backup notebook?

**Colab:**

```
File → Download → Download .ipynb
Atau auto-save ke Google Drive
```

**Jupyter:**

```
File → Download as → Notebook (.ipynb)
```

**Best practice:**

```bash
# Use Git for version control
git init
git add notebook.ipynb
git commit -m "Add training notebook"
git push
```

---

### Q5: Bisa pakai Jupyter di HP/tablet?

**A:** Bisa pakai Google Colab!

1. Buka browser di HP
2. Akses: `colab.research.google.com`
3. Login & buat notebook
4. Coding di HP! 📱

**Limitations:**

- Layar kecil (kurang nyaman)
- Keyboard virtual
- Better pakai tablet + keyboard

---

### Q6: Cara install PyTorch dengan CUDA yang benar?

**A:** Sesuaikan dengan CUDA version:

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch
# CUDA 11.8:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only:
pip3 install torch torchvision torchaudio
```

**Verify:**

```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

---

## 📚 Resources Tambahan

### 📖 Documentation:

- [Jupyter Documentation](https://jupyter.org/documentation)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Anaconda Documentation](https://docs.anaconda.com/)

### 🎥 Video Tutorials:

- "Jupyter Notebook Tutorial for Beginners"
- "Google Colab Complete Tutorial"
- "Setting up Deep Learning Environment"

### 💬 Communities:

- [r/JupyterNotebook](https://www.reddit.com/r/JupyterNotebook/)
- [Google Colab Reddit](https://www.reddit.com/r/GoogleColab/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/jupyter-notebook)

---

## 🎉 Kesimpulan

### Ringkasan Cepat:

| Kebutuhan        | Platform | Alasan                    |
| ---------------- | -------- | ------------------------- |
| Belajar ML       | Kaggle   | GPU gratis kuat + dataset |
| Kerja tim        | Colab    | Real-time collaboration   |
| Production       | Jupyter  | Full control + offline    |
| Data sensitive   | Jupyter  | Privacy                   |
| Eksperimen cepat | Colab    | No setup needed           |

### Rekomendasi Path Belajar:

```
Week 1-2: Kaggle
├─ Pelajari dasar ML
├─ Pakai dataset built-in
└─ GPU gratis 30 jam/minggu

Week 3-4: Google Colab
├─ Belajar kolaborasi
├─ Upload dataset sendiri
└─ Integrasi Google Drive

Month 2+: Jupyter Local
├─ Install di komputer
├─ Customisasi environment
└─ Production-ready workflow
```

---

## 🚀 Quick Start Commands

### Google Colab:

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Check GPU
import torch
print(torch.cuda.is_available())

# Install libraries
!pip install monai nibabel
```

### Jupyter Local:

```bash
# Create environment
conda create -n myenv python=3.10

# Activate
conda activate myenv

# Install Jupyter
pip install jupyter

# Start Jupyter
jupyter notebook
```

---

**Made with ❤️ for ML learners**  
_Last updated: October 22, 2025_

---

**[⬆ Back to Top](#-setup-jupyter-notebook--google-colab-untuk-pemula)**
