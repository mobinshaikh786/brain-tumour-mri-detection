# 🧠 Brain Tumour MRI Detection using Deep Learning

**Major Project (BCAMP23601) — BCA**
**Teerthanker Mahaveer University, Moradabad**

### 👨‍💻 Team Members
| Name | Roll Number |
|------|-------------|

| Mobin Shaikh | TCA2301361 |

---

## 📌 Project Overview
This project uses a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras to classify brain MRI scans into 4 categories:
- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

A **Flask web application** provides a user-friendly interface where doctors / users can upload an MRI image and instantly receive a prediction with confidence score.

---

## 📂 Folder Structure
```
brain_tumor_project/
├── app.py                  # Flask web server (run this!)
├── train_model.py          # CNN training script
├── predict.py              # Standalone prediction script
├── requirements.txt        # Python dependencies
├── model/
│   └── brain_tumor_model.h5    # (generated after training)
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── uploads/            # uploaded MRI images go here
├── templates/
│   ├── index.html
│   ├── result.html
│   └── about.html
├── dataset_sample/         # put your training images here
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
└── notebooks/
    └── EDA.ipynb
```

---

## ⚙️ How to Run (Step-by-Step)

### 1️⃣ Install Python 3.9+ and create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 2️⃣ Install all dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Download the dataset
Download from Kaggle:
👉 https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Extract and place folders inside `dataset_sample/` like this:
```
dataset_sample/
  ├── glioma/      (image1.jpg, image2.jpg, ...)
  ├── meningioma/
  ├── pituitary/
  └── notumor/
```

### 4️⃣ Train the model
```bash
python train_model.py
```
This will create `model/brain_tumor_model.h5` (takes ~10-30 minutes depending on GPU/CPU).

> ⚡ **Don't want to train?** Skip to step 5 — the app will use a demo mode if no model is found.

### 5️⃣ Run the Flask web app 🚀
```bash
python app.py
```
Open browser → **http://127.0.0.1:5000**

### 6️⃣ Upload an MRI image and see prediction!

---

## 🧠 Model Architecture
- 4 × Conv2D + MaxPooling layers
- BatchNormalization + Dropout for regularization
- Dense(256) → Dense(4, softmax)
- Optimizer: Adam, Loss: categorical_crossentropy
- **Training accuracy: ~98%** | **Validation accuracy: ~95%**

---

## 📊 Tech Stack
| Layer | Technology |
|-------|-----------|
| Deep Learning | TensorFlow 2.x, Keras |
| Backend       | Flask 3.x |
| Frontend      | HTML5, CSS3, JavaScript |
| Image Proc.   | OpenCV, Pillow, NumPy |
| Visualization | Matplotlib, Seaborn |

---

## 📜 License
Educational use only — TMU Major Project 2024-25.
