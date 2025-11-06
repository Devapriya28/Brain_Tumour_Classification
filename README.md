# 🧠 Brain Tumor MRI Image Classification

A **Streamlit web application** that classifies **brain MRI images** into different tumor types using a **deep learning model** trained on medical imaging data.  
The goal of this project is to assist in early tumor detection through image-based prediction.

---

## 💡 Project Overview

This project demonstrates how deep learning can be applied in healthcare diagnostics.  
By uploading a brain MRI image, the model predicts which tumor category the image belongs to.

### 🔍 Tumor Categories:
- Glioma  
- Meningioma  
- Pituitary  
- No Tumor  

---

## ⚙️ Tech Stack

| Category | Tools Used |
|-----------|-------------|
| **Programming Language** | Python 🐍 |
| **Frameworks** | TensorFlow, Streamlit |
| **Libraries** | NumPy, Pillow, Pickle |
| **Model Type** | InceptionV3 (Pretrained CNN) |
| **Interface** | Streamlit Web App |

---

## 🚀 How It Works

1. **Upload** a brain MRI image (JPG/PNG).  
2. The image is **preprocessed** (resized, normalized).  
3. The trained deep learning model **predicts** the tumor type.  
4. The app displays:  
   - Predicted class  
   - Confidence score  
   - Bar chart of probabilities  

---

## 🧠 Model Information

- Model Name: `MobileNetV2_best.pkl`  
- Input Size: 224x224 pixels  
- Framework: TensorFlow  
- Accuracy: ~95% (based on test dataset)

---

## 💻 How to Run the App

1. Clone or copy this project to your system.  
2. Open in VS Code or any Python IDE.  
3. Install the required libraries:
   ```bash
   pip install streamlit tensorflow numpy pillow pickle-mixin
