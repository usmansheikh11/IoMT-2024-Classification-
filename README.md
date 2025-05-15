# IoMT-2024 Dataset Classification Analysis

This repository contains the code and results for the classification analysis of the **CICIoMT2024** dataset, focusing on cybersecurity attack detection in Internet of Medical Things (IoMT) networks.

## Project Overview

- **Dataset:** CICIoMT2024 — traffic data from 40 IoMT devices under benign and various attack conditions.
- **Tasks:** Binary classification (Benign vs Attack) and multiclass classification (Benign, DDoS, DoS, Recon, Spoofing, MQTT).
- **Models:** Decision Tree, Random Forest, SVM, and a Deep Learning Neural Network (Multilayer Perceptron).
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
- **Key Findings:** The neural network achieved the highest accuracy and F1-score across tasks, while Random Forest offers good interpretability with competitive performance.

## Repository Structure

- `data/` - Contains the dataset (or links to dataset source).
- `src/` - Python scripts implementing data processing, model training, and evaluation.
- `notebooks/` - Jupyter notebooks for exploratory data analysis or experiments.
- `models/` - Saved model files.
- `results/` - Evaluation results and visualizations.
- `requirements.txt` - Required Python packages.
- `README.md` - This file.

## Getting Started

### Prerequisites

- Python 3.8+
- Install dependencies:

```bash
pip install -r requirements.txt
