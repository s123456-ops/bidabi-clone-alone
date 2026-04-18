# Modular_Translation_Pipeline – Data Engineering & ML Project

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

## 📊 Detailed Pipeline Implementation

### 1. Data Ingestion (RAW) - `asyscrapper.py`

**Purpose:** Asynchronously scrape product data from Open Food Facts API for multiple categories.

**Key Features:**
- Asynchronous HTTP requests using `aiohttp` for high performance
- Robust error handling with exponential backoff retry logic
- Concurrent image downloading with semaphore limits to avoid overwhelming the server
- Product validation (must have ID, name, categories, and image)
- CSV export with structured metadata

**Code Highlights:**
```python
# Asynchronous API fetching with retry logic
async def fetch_page(session, category, page, page_size, sem):
    async with sem:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                await asyncio.sleep(REQUEST_DELAY)
                async with session.get(API_URL, params=params) as resp:
                    data = await resp.json()
                    products = data.get("products", [])
                    return products
            except Exception as e:
                # Exponential backoff retry
                wait_time = RETRY_DELAY * (2 ** attempt)
                await asyncio.sleep(wait_time)
```

**Challenges Faced:**
- API rate limiting and 503 errors required implementing retry logic
- Handling missing or invalid product data
- Managing concurrent requests without getting banned
- Dealing with various image URL formats and fallbacks

**Run the scraper:**
```bash
python src/asyscrapper.py
```

**Output:** Creates `data/raw/metadata_<category>_180.csv` and downloads images to `data/raw/images/<category>/`

### 2. Data Processing (INTERIM → PROCESSED) - `data_processor.py`

**Purpose:** Split raw images into train/validation/test sets for ML training.

**Key Features:**
- Reproducible data splitting using scikit-learn (60/20/20 split)
- Automatic folder structure creation
- File copying with metadata preservation
- Random seed for consistency

**Code Highlights:**
```python
# Reproducible splitting
train_imgs, temp_imgs = train_test_split(images, test_size=0.4, random_state=42)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
```

**Run data processing:**
```bash
python src/data_processor.py
```

**Output:** Organized images in `data/processed/train/`, `val/`, `test/` folders

### 3. Model Training & Evaluation - `classificator.py`

**Purpose:** Train a ResNet18 classifier with advanced techniques and comprehensive evaluation.

**Key Features:**
- Full fine-tuning of pre-trained ResNet18
- MixUp data augmentation for better generalization
- Early stopping with patience to prevent overfitting
- Extensive evaluation: confusion matrix, ROC curves, per-class accuracy
- Embedding visualization with t-SNE and UMAP
- Hardest samples analysis

**Model Architecture:**
```python
def create_resnet18(num_classes):
    model = resnet18(weights="IMAGENET1K_V1")
    # Full fine-tuning (all layers trainable)
    for param in model.parameters():
        param.requires_grad = True
    
    # Custom classifier head
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    return model
```

**Training Augmentations:**
- Random horizontal flip, rotation, affine transforms
- Color jittering, Gaussian blur
- MixUp augmentation during training

**Evaluation Metrics:**
- Classification report (precision, recall, F1-score)
- Confusion matrix heatmap
- Per-class accuracy bar chart
- ROC curves with AUC scores
- t-SNE and UMAP embeddings visualization
- Hardest samples identification

**Challenges Faced:**
- Balancing model complexity with limited data (~180 images per class)
- Implementing MixUp correctly for multi-class classification
- Handling class imbalance in product categories
- Optimizing hyperparameters (learning rate, batch size, epochs)

**Run training:**
```bash
python src/classificator.py
```

**Output:** Trained model saved as `models/best_model_resnet18_finetuned.pth`

---

## 🤖 Model Details

- **Architecture:** ResNet18 (pre-trained on ImageNet)
- **Fine-tuning:** Full fine-tuning of all layers
- **Input:** 256x256 RGB images (resized from various sizes)
- **Classes:** bread, butter, champagnes, milk, sugar (5 classes)
- **Training:** 20 epochs max, early stopping with patience=3
- **Augmentation:** MixUp (α=0.4), random flips, rotations, color jitter
- **Optimizer:** Adam (lr=1e-5, weight_decay=1e-4)
- **Scheduler:** Cosine annealing
- **Batch Size:** 32

**Performance Metrics (Example Results):**
- Validation Accuracy: ~85-90% (varies by run)
- Best categories: milk, bread
- Challenging categories: butter, champagnes (similar appearances)

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

## 📈 Usage & Inference

### Running the Full Pipeline

1. **Scrape data:**
   ```bash
   python src/asyscrapper.py
   ```

2. **Process data:**
   ```bash
   python src/data_processor.py
   ```

3. **Train model:**
   ```bash
   python src/classificator.py
   ```

### Inference on New Images

```python
import torch
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn

# Load model
model = resnet18()
model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, 5))
model.load_state_dict(torch.load('models/best_model_resnet18_finetuned.pth'))
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Predict
image = transform(your_image).unsqueeze(0)
with torch.no_grad():
    output = model(image)
    predicted_class = torch.argmax(output, dim=1).item()

classes = ['bread', 'butter', 'champagnes', 'milk', 'sugar']
print(f"Predicted: {classes[predicted_class]}")
```

---

## 🧠 Key Learnings & Challenges

### Technical Learnings
- **Asynchronous Programming:** Using `aiohttp` and semaphores for efficient web scraping
- **Data Pipeline Design:** Modular approach with clear RAW → PROCESSED stages
- **ML Engineering:** Transfer learning, data augmentation, regularization techniques
- **Model Evaluation:** Comprehensive metrics beyond accuracy (ROC, confusion matrix)
- **Version Control:** Separating code (Git) from data/models (DVC)

### Challenges Overcome
- **API Reliability:** Implemented robust retry logic with exponential backoff
- **Data Quality:** Added validation for product completeness and image availability
- **Concurrency Management:** Used semaphores to limit concurrent requests/images
- **Model Training:** Balanced overfitting with early stopping and regularization
- **Class Imbalance:** Some categories had fewer valid products than others

### Issues Encountered
- Initial synchronous scraper (`data_loader.py`) was too slow for large datasets
- API timeouts required increasing timeout values and retry attempts
- Some product images were corrupted or unavailable
- Training script had dependency issues (UMAP optional, matplotlib display)

---

## 📄 Dependencies

Key packages from `requirements.txt`:
- `aiohttp==3.13.5` - Async HTTP client
- `torch==2.11.0` - PyTorch deep learning
- `torchvision==0.26.0` - Computer vision utilities
- `scikit-learn==1.8.0` - ML metrics and splitting
- `matplotlib==3.10.8` - Plotting
- `seaborn==0.13.2` - Statistical visualization
- `umap-learn==0.5.6` - Dimensionality reduction
- `numpy==2.4.4` - Numerical computing

---

## 📊 Dataset Statistics

- **Total Categories:** 5 (bread, butter, champagnes, milk, sugar)
- **Images per Category:** ~180 raw images
- **Train/Val/Test Split:** 60%/20%/20%
- **Total Training Images:** ~540
- **Image Resolution:** Variable (resized to 256x256 for training)
- **Data Source:** Open Food Facts API (crowdsourced product database)

---

## 🔬 Future Improvements

- Implement data augmentation pipeline for better generalization
- Add more categories or handle variable number of classes
- Deploy model as web service (FastAPI/Flask)
- Add model monitoring and retraining pipeline
- Experiment with other architectures (EfficientNet, Vision Transformers)
- Add automated hyperparameter tuning

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

