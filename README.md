# UPI Fraud Detection

## Overview
This repository contains an AI/ML-based solution for detecting fraudulent transactions on a UPI-scale dataset. The goal is to develop a robust and scalable model capable of identifying anomalies, suspicious patterns, and fraudulent activities in real-time or through post-transaction analysis.

## Dataset Information
This dataset consists of two files: [link to dataset](https://drive.google.com/drive/folders/1jw1_nJ_HOZC0jr--O9r2uCL0bMzslzzx?usp=drive_link)
- **Transaction Data**: 2.3 GB (CSV format)
- **MCC (Merchant Category Code) Data**: 99 KB (CSV format)

### Key Observations
#### **1. Transaction Data**
- **Total Transactions**: 24,386,900
- **Fraudulent Transactions**: 388,431 (~1.6% of dataset)
- **Merchants & Locations**:
  - **98,953 unique merchants**
  - **13,429 unique merchant cities**
- **Most Common Payment Method**: Swipe Transaction (15.4M times)

#### **2. MCC (Merchant Category Code) Data**
- **High-Risk MCC Categories**: Certain MCCs show a higher correlation with fraudulent transactions.
- **Most Common MCC Category**: "Airlines" (128 occurrences).
- **Fraud-Prone MCCs**: Some business categories, such as cash disbursement and online services, show an increased likelihood of fraud.
- **IRS Reportable Transactions**: 83% of transactions fall under reportable categories.

### **MCC & Transaction Statistics (Comparison)**
| Metric                     | MCC Data                 | Transaction Data          |
|----------------------------|--------------------------|---------------------------|
| Total Records              | 981                      | 24,386,900                |
| Key Column                 | `mcc`                    | `transaction_id`          |
| Most Common Category       | "Airlines" (128 times)  | "Swipe Transaction" (15.4M) |
| Fraud-Prone MCC Categories | Identified and categorized | 1.6% fraudulent cases     |
| Reportable Transactions    | 83%                      | Significant fraud cases observed |

## Scaling & Optimization
To efficiently handle large-scale data, we converted CSV files to Parquet format, which offers:
- **Better Compression**: Reduces storage size significantly.
- **Faster Read/Write Operations**: Speeds up data processing.
- **Optimized Query Performance**: Ideal for large datasets used in ML models.

#### **Conversion Process:**
```python
import pandas as pd

# Load CSV and convert to Parquet
transaction_df = pd.read_csv('transactions.csv')
transaction_df.to_parquet('transactions.parquet', engine='pyarrow', compression='snappy')
```

## Model Approach
### **1. Preprocessing Steps**
- Label encoding for categorical variables
- Conversion of `Amount` to numeric format
- Timestamp feature extraction (hour, weekday, weekend indicators)
- Handling missing values in MCC and ZIP codes
- Feature engineering based on:
  - Transaction frequency
  - Time of transaction
  - Merchant patterns
  - Declined transactions before a fraud event

### **2. Feature Selection & Engineering**
- **Time-based features**: Hour, day of the week, weekend vs. weekday transactions.
- **Amount patterns**: Identifying frequent high-value transactions.
- **Merchant-based patterns**: Identifying merchants with repeated fraudulent activity.
- **MCC Analysis**: Grouping transactions based on high-risk categories.

### **3. Modeling Techniques**
- **Supervised Learning**:
  - XGBoost
- **Unsupervised Learning**:
  - Isolation Forest

### **4. Handling Imbalanced Data**
Fraud cases constitute only **1.6%** of the dataset, making the data highly imbalanced. To address this:
- **Resampling Techniques**:
  - **Oversampling**: Using SMOTE to synthetically generate fraud samples.
  - **Undersampling**: Reducing non-fraud transactions to balance the dataset.
- **Cost-Sensitive Learning**:
  - Assigning higher misclassification penalties for fraud cases in XGBoost using `scale_pos_weight`.
- **Anomaly Detection Approach**:
  - Using **Isolation Forest**, which is robust to class imbalance.
- **Evaluation Metrics Beyond Accuracy**:
  - **Precision & Recall Trade-off** to reduce false negatives.
  - **ROC-AUC & PR-AUC** for better fraud detection performance.

## Architecture & Deployment
### **System Architecture**
1. **Data Ingestion**: Raw transactions are collected and stored in a Parquet data warehouse.
2. **Feature Engineering & Processing**: Extracting relevant features using Pandas and NumPy.
3. **Model Training**:
   - Supervised learning (XGBoost) and unsupervised learning (Isolation Forest).
4. **Real-Time Detection with FastAPI**:
   - API receives transaction details and runs predictions.
5. **Alert & Action System**:
   - Flags suspicious transactions and triggers alerts.

### **Deployment with FastAPI**
For real-time fraud detection, we use **FastAPI**, a high-performance web framework.

## Framework & Tools Used
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-Learn, XGBoost
- **Anomaly Detection**: Isolation Forest
- **Deployment**: FastAPI for real-time fraud detection API
- **Big Data Processing**: Spark (if scaling to large datasets)


