# 🛒 Product Classification Pipeline – Data Engineering & ML Project

## 📌 Overview

This project implements a **complete machine learning pipeline** for product classification, including data collection, preprocessing, model training, and inference. It scrapes product data from the Open Food Facts API, processes images and metadata, and trains a ResNet18-based classifier for categories like bread, butter, champagnes, milk, and sugar.

It addresses real-world challenges such as:
- Handling unreliable APIs (e.g., 503 errors)
- Structuring raw data consistently
- Ensuring dataset reproducibility with DVC
- Separating code versioning (Git) from data versioning (DVC)
- Building and deploying ML models

The pipeline follows a standard ML architecture:
RAW → INTERIM → PROCESSED → MODEL TRAINING → INFERENCE

---

## 🎯 Objectives

- Build a scalable data ingestion and processing pipeline
- Apply best practices in data organization and ML engineering
- Ensure full reproducibility of datasets and models
- Prepare data for analytics and machine learning use cases
- Train and deploy a product classification model

---

## 🧰 Tech Stack

- **Python** (Core language)
- **aiohttp** (Asynchronous API requests)
- **PyTorch** (Deep learning framework)
- **torchvision** (Computer vision utilities)
- **Pandas** & **NumPy** (Data manipulation)
- **Git & GitHub** (Code versioning)
- **DVC** (Data and model versioning)
- **PIL/Pillow** (Image processing)

---

## 📁 Project Structure

```
bidabi-clone-alone/
│
├── data/
│   ├── raw/
│   │   ├── images/<category>/          # Raw product images
│   │   └── metadata_<category>_<count>.csv  # Raw metadata
│   ├── interim/                        # Intermediate processed data
│   ├── processed/                      # Final processed datasets
│   │   ├── train/<category>/           # Training images
│   │   ├── val/<category>/             # Validation images
│   │   └── test/<category>/            # Test images
│   └── localstore/                     # DVC cache
│
├── models/
│   ├── best_model_resnet18_finetuned.pth  # Trained model
│   └── best_model_resnet18_finetuned.pth.dvc  # Model versioning
│
├── src/
│   ├── asyscrapper.py                  # Data scraping script
│   ├── data_processor.py               # Data preprocessing
│   ├── data_loader.py                  # PyTorch data loaders
│   └── classificator.py                # Model training and inference
│
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── LICENSE                             # Project license
```

---

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/bidabi-clone-alone.git
   cd bidabi-clone-alone
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize DVC:**
   ```bash
   dvc init
   dvc pull  # If data is stored remotely
   ```

---

## 📊 Data Pipeline

### 1. Data Ingestion (RAW)
- Data collected from [Open Food Facts API](https://world.openfoodfacts.org/data)
- Asynchronous scraping using `aiohttp`
- Images downloaded per category
- Metadata stored in CSV files

**Run the scraper:**
```bash
python src/asyscrapper.py
```

### 2. Data Processing (INTERIM → PROCESSED)
- Image preprocessing and augmentation
- Dataset splitting (train/val/test)
- Metadata cleaning and structuring

**Process the data:**
```bash
python src/data_processor.py
```

### 3. Model Training
- Fine-tuned ResNet18 architecture
- Transfer learning approach
- Training on processed datasets

**Train the model:**
```bash
python src/classificator.py
```

---

## 🤖 Model Details

- **Architecture:** ResNet18 (pre-trained on ImageNet)
- **Fine-tuning:** Last layers adapted for 5-class classification
- **Input:** 224x224 RGB images
- **Classes:** bread, butter, champagnes, milk, sugar
- **Performance:** [Add metrics if available, e.g., accuracy, F1-score]

The trained model is saved as `models/best_model_resnet18_finetuned.pth`

---

## 🔄 Data & Model Versioning (DVC)

- Git tracks code and `.dvc` files
- DVC tracks actual datasets and models
- Ensures reproducibility across environments

**Track changes:**
```bash
dvc add data/processed
dvc add models/best_model_resnet18_finetuned.pth
git add .
git commit -m "Update data and model versions"
```

---

## 📈 Usage

### Inference on New Images
```python
from src.classificator import ProductClassifier

classifier = ProductClassifier(model_path='models/best_model_resnet18_finetuned.pth')
prediction = classifier.predict('path/to/image.jpg')
print(f"Predicted class: {prediction}")
```

### Reproducing the Pipeline
1. Run data scraping
2. Process data
3. Train model
4. Evaluate on test set

---

## 🧠 Key Learnings

- Modular data pipeline design
- Data vs. code versioning (Git vs. DVC)
- Handling real-world API limitations
- Computer vision model fine-tuning
- Structuring scalable ML projects

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👩‍💻 Author

[Your Name] - [Your Contact/LinkedIn]

---

*Built with ❤️ for data engineering and ML enthusiasts*
│ │ └── metadata_<category>_<count>.csv
│ ├── interim/
│ ├── processed/
│ └── localstore/
│
├── src/
│ └── asyscrapper.py
│
├── raw.dvc
├── .gitignore
├── README.md
├── requirements.txt

---

## ⚙️ Data Pipeline

### Data Ingestion (RAW)

- Data collected from Open Food Facts API  
- Images downloaded per category  
- Metadata stored in CSV files  

**Example:**

data/raw/images/milk/
data/raw/metadata_milk_180.csv


---

## 🔄 Data Versioning (DVC)

- Git tracks code and `.dvc` files  
- DVC tracks actual datasets  

---

📊 Dataset

Example categories: sugar, milk, bread

Each category includes:
product images and metadata (CSV)

---

🧠 Key Learnings
- Modular data pipeline design
- Data vs code versioning (Git vs DVC)
- Handling real-world API limitations
- Structuring scalable projects

---

👩‍💻 Author
Salma ALAOUI MRANI

