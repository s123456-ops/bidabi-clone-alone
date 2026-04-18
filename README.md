# Food Product Classification Pipeline

A data engineering and machine learning project for classifying food products using images and metadata from the Open Food Facts API.

---

## 📌 Overview

This project implements a complete ML pipeline:

* data ingestion from an external API
* preprocessing and dataset structuring
* model training using deep learning
* evaluation and inference

The pipeline follows a standard architecture:

**RAW → INTERIM → PROCESSED → TRAINING → INFERENCE**

---

## 🎯 Objectives

* Build a scalable data pipeline
* Structure datasets for ML workflows
* Train a classification model on real-world data
* Ensure reproducibility using DVC
* Apply best practices in ML engineering

---

## 🧰 Tech Stack

* Python
* PyTorch & torchvision
* aiohttp (async scraping)
* Pandas & NumPy
* DVC (data & model versioning)
* Git & GitHub

---

## 🧱 Project Structure

```text
data/
├── raw/
├── interim/
├── processed/

models/
├── best_model_resnet18_finetuned.pth

src/
├── asyscrapper.py
├── data_processor.py
├── data_loader.py
├── classificator.py
```

---

## ⚙️ Pipeline Steps

### 1. Data Ingestion

* Asynchronous scraping using `aiohttp`
* Retry logic for API errors (e.g., 503)
* Image download + metadata storage

### 2. Data Processing

* Train / validation / test split (60/20/20)
* Reproducible dataset preparation

### 3. Model Training

* ResNet18 (pretrained on ImageNet)
* Fine-tuning all layers
* Data augmentation (MixUp, transforms)
* Early stopping

### 4. Evaluation

* Accuracy, precision, recall, F1-score
* Confusion matrix and ROC curves

---

## 🤖 Model Details

* Architecture: ResNet18
* Classes: bread, butter, champagnes, milk, sugar
* Input: 256×256 images
* Optimizer: Adam
* Scheduler: cosine annealing

---

## 🔄 Reproducibility

* Git → code versioning
* DVC → data and model versioning

---

## 🚀 Usage

```bash
python src/asyscrapper.py
python src/data_processor.py
python src/classificator.py
```

---

## 🧠 Key Learnings

* Designing modular data pipelines
* Handling unreliable APIs
* Managing data vs model versioning
* Applying deep learning to real-world data

---

## 🔬 Future Improvements

* Add more categories
* Deploy as API (FastAPI)
* Experiment with other architectures
* Automate hyperparameter tuning

---

## 👤 Author

Salma ALAOUI MRANI
