# 😷 Face Mask Detection Using Transfer Learning

## 📌 Project Overview

This project focuses on **binary image classification for face mask detection**, using transfer learning with three pretrained Convolutional Neural Networks (CNNs): VGG16, InceptionV3, and ResNet50.

The objective is to classify images into two categories:
- **With Mask**
- **Without Mask**

## 🧠 Methodology

The project explores two approaches:

1. **Feature Extraction:** Use pretrained CNNs as frozen feature extractors, then train classical machine learning classifiers (SVM, Random Forest, Decision Tree, and XGBoost).
2. **Fine-Tuning:** Fine-tune VGG16, InceptionV3, and ResNet50 using different freezing rates (0%, 25%, 50%, 75%, and 100%).

Models are evaluated using Accuracy, Precision, Recall, F1-score, ROC-AUC, and confusion matrices.

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- VGG16, InceptionV3 & ResNet50
- Scikit-learn
- XGBoost
- NumPy & Pandas
- Matplotlib & Seaborn

## 📊 Results & Analysis

The project compares feature extraction and fine-tuning approaches to identify the best-performing architecture and freezing rate. It also includes training curves, performance comparisons, confusion matrices, and ROC curves.

## 🖼️ Project Poster

A dedicated poster is also included in the repository, presenting the project's methodology, experiments, and main findings.

## 🚀 Getting Started

1. Open the notebook in Kaggle.
2. Add the Face Mask Dataset.
3. Enable GPU acceleration (recommended).
4. Run the notebook cells sequentially to reproduce the experiments.

## 🎯 Objective

To investigate how transfer learning can improve image classification performance and compare different pretrained CNN architectures for face mask detection.
