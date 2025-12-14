# 🌿 RGB Image-Based Disease Detection in Plants with Computer Vision

This project focuses on detecting plant diseases using RGB images
captured by standard cameras. By leveraging modern deep learning
techniques and multiple CNN architectures, the system classifies leaf
diseases across several crop species with high accuracy.
The work was conducted as part of the **Programming Integration Project
(CO3101)** at **Ho Chi Minh City University of Technology**.

## Video presentation

Our group's presentation can be viewed here: https://www.youtube.com/watch?v=JwhVhIja_9M

The slides we used for our presentation can be view here: https://www.canva.com/design/DAG7Nmcjac8/0vaQe_hqMMY-RypLO3plRA/view?utm_content=DAG7Nmcjac8&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h86f9c836a9

## 📌 Project Information

**Project Title:** RGB Image-Based Disease Detection in Plants with
Computer Vision

**Course:** Programming Integration Project CO3101

**Advisor:** Dr. Nguyễn An Khương

### 👥 Students

-   **Nguyễn Văn An** --- MSSV: 2352013
-   **Hoàng Kim Cương** --- MSSV: 2352145

## 📘 Project Overview

The objective of this project is to develop a deep learning system
capable of identifying common plant diseases from RGB leaf images.
Four CNN models were studied: ResNet-18, MobileNetV1, MobileNetV2, and
VGG-16.

## 📅 Project Plan

https://docs.google.com/document/d/1XKS5BFkYS84xDEkWBfr5Zf_ycowppHS8javq_c1pl3M

## 📚 Dataset

Dataset: PlantVillage
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

Preprocessed dataset used in training:
https://drive.google.com/drive/folders/1MdoFeMnX-bEqPNPBgCnyrBtwGX4xHtdK

## 🤖 Training Details

### Models Implemented

-   ResNet-18
-   MobileNetV1
-   MobileNetV2
-   VGG-16 (not fully trained due to GPU memory limitations)

### Training Configuration

-   Optimizer: AdamW
-   Learning rate: 0.0002
-   Batch size: 32
-   Weight decay: 0.001
-   Custom weight initialization (Kaiming + LeCun)
-   Early stopping enabled

## 📊 Results

| Model        | Accuracy (%) | F1-score | Size (MB) | Parameters (M) |
|--------------|--------------|----------|-----------|----------------|
| ResNet-18    | 97.23        | 0.9725   | 11.19     | 42.71          |
| MobileNetV1  | 93.01        | 0.9295   | 3.23      | 12.39          |
| MobileNetV2  | 94.78        | 0.9471   | 2.25      | 8.71           |

## 🧩 Discussion

### Strengths

-   High accuracy across all models
-   Efficient augmentation and preprocessing
-   ResNet-18 generalizes best

### Limitations

-   Dataset captured in controlled conditions
-   VGG-16 impossible to train due to GPU memory

### Lessons Learned

-   Model building from scratch
-   Hyperparameter tuning
-   Comparing performance vs. efficiency

## 🧾 Code & Data Availability

Source code: https://github.com/BanAnA9205/CO3101_PIProject_Semester251

Curated dataset:
https://drive.google.com/drive/folders/1MdoFeMnX-bEqPNPBgCnyrBtwGX4xHtdK
