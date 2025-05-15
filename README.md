# **Diabetic Retinopathy Classification using AI**

## 📌 Project Overview  
This project aims to develop an **AI model** for **classifying retinal images** to detect **diabetic retinopathy**.  
The classification is based on images from a **fundus camera** and **fluorescein angiography**.  

The model will automatically identify the disease progression in one of five stages:  
- ✅ **No diabetic retinopathy**  
- ✅ **Mild non-proliferative diabetic retinopathy**  
- ✅ **Moderate non-proliferative diabetic retinopathy**  
- ✅ **Severe non-proliferative diabetic retinopathy**  
- ✅ **Proliferative diabetic retinopathy**  

---

## 📂 Dataset  
The dataset used for training and evaluation is the **Diabetic Retinopathy Resized** dataset, available on Kaggle:  
🔗 [https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized](https://www.kaggle.com/datasets/tanlikesmath/diabetic-retinopathy-resized)  

This dataset contains resized retinal images categorized into five classes, which align with the stages of diabetic retinopathy used in this project.

---

## 🛠 Technologies  
- **Language:** Python 🐍  
- **Libraries:** PyTorch, torchvision, OpenCV, PIL, matplotlib, seaborn, scikit-learn  
- **Interface:** Desktop GUI (Tkinter) 🖥️  

---

## 🤖 AI Model Selection & Implementation  
For the task of diabetic retinopathy classification, Convolutional Neural Networks (CNNs) are utilized. The final model is based on EfficientNet-B2, fine-tuned on preprocessed retinal images.  

**Model highlights:**  
- Input image size: 260×260  
- 5-class classification (retinopathy levels 0–4)  
- Grad-CAM visualizations for interpretability  
- Trained using class weighting, data augmentation, and mixed-precision training  

---

## 📈 Results & Metrics  
All training history, performance metrics, evaluation plots, confusion matrix, and Grad-CAM visualizations are saved in the `results/` folder.  
This includes:
- Accuracy, loss curves
- Per-class precision, recall, and F1-score
- Confusion matrix heatmap
- Class distribution comparison
- Grad-CAM visual examples
- Prediction entropy histogram

---

## 👥 Collaborators  
This project is being developed by:  

[![WinterWollf](https://img.shields.io/badge/GitHub-WinterWollf-181717?logo=github&logoColor=white&style=for-the-badge)](https://github.com/WinterWollf)  
[![PawelS12](https://img.shields.io/badge/GitHub-PawelS12-181717?logo=github&logoColor=white&style=for-the-badge)](https://github.com/PawelS12)  
