# 🛒 Product Data Pipeline & Scraper – Data Engineering Project

## 📌 Overview

This project implements a **reproducible data pipeline** for collecting, structuring, and versioning product data from the Open Food Facts API.

It addresses real-world data engineering challenges such as:
- Handling unreliable APIs (e.g. 503 errors)
- Structuring raw data consistently
- Ensuring dataset reproducibility
- Separating code versioning (Git) from data versioning (DVC)

The pipeline follows a standard architecture:
RAW → INTERIM → PROCESSED


---

## 🎯 Objectives

- Build a scalable data ingestion pipeline  
- Apply best practices in data organization  
- Ensure full reproducibility of datasets  
- Prepare data for analytics and machine learning use cases  

---

## 🧰 Tech Stack

- Python  
- aiohttp (asynchronous API requests)  
- Git & GitHub  
- DVC (Data Version Control)  

---

## 📁 Project Structure
bidabi-clone-alone/
│
├── .dvc/
├── data/
│ ├── raw/
│ │ ├── images/
│ │ │ └── <category>/
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

Example categories:
sugar
milk
bread

Each category includes:
product images
metadata (CSV)

---

🧠 Key Learnings
Modular data pipeline design
Data vs code versioning (Git vs DVC)
Handling real-world API limitations
Structuring scalable projects

---

👩‍💻 Author
Salma ALAOUI MRANI
Master Big Data & Business Intelligence
Université Sorbonne Paris Nord
