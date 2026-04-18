# 🍎 Food Product Classification Pipeline

> A data engineering and machine learning project for classifying food products using images and metadata from the Open Food Facts API.

---

## 📌 Overview

This project implements an end-to-end pipeline that:

* collects product data from an external API
* processes and structures datasets for machine learning
* trains a deep learning model for image classification
* evaluates model performance using multiple metrics

Pipeline architecture:

```
RAW → INTERIM → PROCESSED → TRAINING → INFERENCE
```

---

## 🎯 Objectives

* Build a modular and scalable data pipeline
* Structure datasets for reproducible ML workflows
* Train a classification model on real-world data
* Apply best practices in ML engineering and versioning

---

## 🧰 Tech Stack

* **Python**
* **PyTorch & torchvision**
* **aiohttp** (asynchronous data collection)
* **Pandas & NumPy**
* **DVC** (data & model versioning)
* **Git & GitHub**

---

## 🧱 Project Structure

```text
data/
├── raw/
├── interim/
├── processed/

models/
└── best_model_resnet18_finetuned.pth

src/
├── asyscrapper.py
├── data_processor.py
├── data_loader.py
└── classificator.py
```

---

## ⚙️ Pipeline

### 1. Data Ingestion

* Asynchronous scraping using `aiohttp`
* Retry logic for API errors (e.g. 503)
* Image download + metadata extraction

### 2. Data Processing

* Train / validation / test split (60/20/20)
* Reproducible dataset preparation

### 3. Model Training

* ResNet-18 (fine-tuned)
* Data augmentation (MixUp, transformations)
* Early stopping

### 4. Evaluation

* Precision, recall, F1-score
* Confusion matrix
* ROC curves

---

## 🤖 Model

* **Architecture:** ResNet-18
* **Input:** 256×256 images
* **Classes:** bread, butter, champagnes, milk, sugar
* **Validation accuracy:** ~74–90% (depending on run)

---

## 🔄 Reproducibility

This project separates versioning responsibilities:

* **Git** → source code
* **DVC** → datasets and trained models

### Setup

```bash
git clone https://github.com/s123456-ops/ml-translation-pipeline.git
cd ml-translation-pipeline

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

dvc pull
```

---

## 🚀 Usage

Run the full pipeline:

```bash
python src/asyscrapper.py
python src/data_processor.py
python src/classificator.py
```

---

## 🧠 Key Learnings

* Designing modular data pipelines
* Handling unreliable external APIs
* Managing data vs model versioning (Git + DVC)
* Applying deep learning to real-world datasets

---

## 🔬 Future Improvements

* Add more product categories
* Experiment with alternative architectures
* Deploy the model as an API
* Automate training and evaluation

---

## 👤 Author

Salma ALAOUI MRANI
