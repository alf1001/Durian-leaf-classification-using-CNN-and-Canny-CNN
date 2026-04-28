# 🌿 Durian Leaf Classification using CNN and Canny-CNN

A web-based application for classifying durian leaf types using **Convolutional Neural Network (CNN)** and **Canny Edge Detection + CNN (Canny-CNN)**, built with **Streamlit**.

---

## 📌 Overview

This project aims to compare two approaches in image classification:

* 🧠 **CNN (RGB Image Only)**
* ⚡ **Canny-CNN (RGB + Edge Detection)**

The system processes leaf images and predicts the durian variety with confidence scores.

---

## 🎯 Features

* Upload durian leaf images
* Automatic leaf detection & cropping
* Edge detection using Canny
* Dual model prediction:

  * CNN (RGB)
  * Canny-CNN (4-channel input)
* Top-3 prediction results
* Confidence score visualization
* Interactive web interface using Streamlit

---

## 🧠 Model Architecture

### CNN (RGB)

* Input: 224 × 224 × 3
* Convolution layers + BatchNorm + MaxPooling
* Global Average Pooling
* Dense + Dropout
* Softmax output (6 classes)

### Canny-CNN

* Input: 224 × 224 × 4 (RGB + Edge)
* Same architecture as CNN
* Additional edge channel improves feature extraction

---

## 🌿 Classes

The model classifies 6 types of durian leaves:

* Bawor
* Duri Hitam
* Matahari
* Montong
* Musang King
* Super Tembaga

---

## 🛠️ Tech Stack

* Python
* Streamlit
* TensorFlow / Keras
* OpenCV
* NumPy
* Pandas
* Pillow

---

## 📂 Project Structure

```
Durian-leaf-classification/
│
├── app.py
├── requirements.txt
├── best_model_fold_4.keras
├── best_model.keras
├── README.md
```

---

## ⚙️ Installation (Local)

```bash
git clone https://github.com/alf1001/Durian-leaf-classification-using-CNN-and-Canny-CNN.git
cd Durian-leaf-classification-using-CNN-and-Canny-CNN

pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deployment

This app can be deployed easily using
👉 Streamlit Community Cloud

Steps:

1. Push project to GitHub
2. Login to Streamlit Cloud
3. Create new app
4. Select repo & `app.py`
5. Deploy 🚀

---

## ⚠️ Notes

* Ensure model files (`.keras`) are included in the repository
* Do not upload virtual environments (`tf-env`, `venv`)
* Input images should be similar to training data for best accuracy

---

## 📊 Output Example

* Predicted class
* Confidence score
* Top-3 probabilities
* Visualization chart

---

## 👨‍💻 Author

**Ahmad Alfin**
Durian Leaf Classification Research Project

