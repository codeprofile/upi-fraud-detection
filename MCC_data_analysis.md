### **MCC (Merchant Category Code) Analysis**

#### **1️⃣ Overview of the Data**
- **Dataset Size**: 981 rows.
- **Primary Column**: `mcc` (Merchant Category Code) – a numeric identifier for business categories.
- **Categorical Columns**:
  - `edited_description`
  - `combined_description`
  - `usda_description`
  - `irs_description`
  - `irs_reportable`
- **Missing Values**:
  - `usda_description`: 275 missing values (706 non-null).
  - `combined_description`: 8 missing values (973 non-null).

---

#### **2️⃣ Key Statistics for MCC Codes**
- **Mean MCC Code**: 4265.23 (wide range of MCCs).
- **Minimum MCC**: 742.
- **Maximum MCC**: 9950.
- **Standard Deviation**: 1616.14 (moderate spread in MCC distribution).
- **Percentiles**:
  - 25th percentile: 2331.
  - 50th percentile (Median): 3586.
  - 75th percentile: 5094.
- **Insight**: Most MCCs lie between 2331 and 5094, with some high-value outliers.

---

#### **3️⃣ Categorical Columns Analysis**
- **Most Common Business Category (`edited_description`)**:
  - **"Airlines"**: Appears 128 times (most frequent).
- **Most Common Combined Description (`combined_description`)**:
  - **"Airlines"**: Also the top category.
- **Most Common `usda_description`**:
  - **"Financial Institutions – Manual Cash Disbursement"**: Appears twice (most other categories are unique).
- **IRS Reporting (`irs_reportable`)**:
  - **963 non-null values** out of 981.
  - **810 transactions are IRS-reportable** (83% of data).
  - **5 unique values**: Likely binary classification (e.g., Yes/No or categories).

---

#### **4️⃣ Missing Data Concerns**
- **`usda_description`**:
  - 275 missing values (706 non-null).
  - **Action**: Impute with `"Unknown"` or analyze impact on findings.
- **`combined_description`**:
  - 8 missing values (973 non-null).
  - **Action**: Impute with `"Unknown"` or investigate patterns.

---

#### **5️⃣ Insights and Recommendations**
- **High-Frequency MCCs**:
  - Focus on MCCs like **"Airlines"** for deeper analysis (e.g., fraud patterns, transaction trends).
- **IRS-Reportable Transactions**:
  - 83% of transactions are IRS-reportable – ensure compliance and reporting mechanisms.
- **Missing Data**:
  - Address missing values in `usda_description` and `combined_description` to avoid bias in analysis.
- **Outliers**:
  - Investigate high-value MCCs (e.g., 9950) for potential anomalies or special categories.

