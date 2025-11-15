[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/R05VM8Rg)
# IIT-Madras-DA2401-Machine-Learning-Lab-End-Semester-Project

## 📌 Purpose of this Template

This repository is the **starter** for your End Semester Project submission in GitHub Classroom. You can implement your solution and push your work in this repository. Please free to edit this README.md file as per your requirements.

> **Scope (as per assignment brief):**  
> This project implements a complete multi-class classification system for the MNIST handwritten digit dataset using only plain Python and basic numerical libraries (NumPy/SciPy). The goal is to apply and extend the machine-learning concepts learned in class, including Linear Regression, Classification, Clustering, Anomaly Detection, and ensemble methods such as Bagging and Boosting.  
>
> The MNIST dataset (train: 10,002 entries, validation: 2,499 entries) consists of 28×28 grayscale digit images represented as 784 pixel-intensity values. Using these datasets, we train multiple models coded entirely from scratch, tune their hyperparameters, analyze bias-variance trade-offs, and evaluate performance using the F1 score.  
>
> The final system combines multiple algorithms and ensemble techniques to achieve improved accuracy, while keeping total training time under 5 minutes. The trained models will later be evaluated by TAs on a hidden MNIST test set to assess both accuracy and runtime.

---

**Important Note:** 
1. TAs will evaluate using the `.py` file only.
2. All your reports, plots, visualizations, etc pertaining to your solution should be uploaded to this GitHub repository

---

## 📁 Repository Structure


- **algorithms.py** — contains all algorithm implementations written from scratch.  
- **main.py** — loads MNIST CSV data, preprocesses it, trains all models, performs ensembling, and prints final results.  
- **ENDSEM_LAB_REPORT.pdf** — final written report submitted for evaluation.  
- **README.md** — project documentation.

---



## 📦 Installation & Dependencies
```
pip install numpy pandas scikit-learn
```
---

## ▶️ Running the Code

All experiments should be runnable from the command line **and** reproducible in the notebook.

### A. Command-line (recommended for grading)

Before running the model, please ensure the following:

- run `algorithms.py` contains **all algorithm implementations**  
  Make sure this file is present in the same directory.

- run  `main.py` : 
Make sure you do Data preprocessing steps-
1. **Normalize** the raw MNIST pixel values (divide by 255.0)  
2. **Standardize**
3. **Apply PCA** (128 components) 




## 🧾 Authors

**< Rathod Shruthi DA24B022 >**, IIT Madras (2025–26)


## Best Practices:
* Keep commits with meaningful messages.
* Please do not write all code on your local machine and push everything to GitHub on the last day. The commits in GitHub should reflect how the code has evolved during the course of the assignment.
* Collaborations and discussions with other students is strictly prohibited.
* Code should be modularized and well-commented.

