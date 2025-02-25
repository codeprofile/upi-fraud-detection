### **Transactional Dataset Descriptive Analysis**

---

## **1. Dataset Overview**
- **Total Rows**: 24,386,900 transactions.  
- **Time Range**: 1991-2020 (potential outliers in older transactions).  
- **Fraudulent Transactions**: 388,431 (~1.6% of the dataset).  
- **Merchant Diversity**: 98,953 unique merchants and 13,429 unique merchant cities.  
- **Payment Methods**: Dominated by "Swipe Transactions" (15.4M occurrences).  

---

## **2. Date & Time Analysis**
- **Time Column**: 1,440 unique values, indicating full 24-hour cycle tracking.  
- **Mean Transaction Date**: Around 2012-06-15.  
- **Minimum Transaction Date**: 1991 (likely incorrect or system-generated test data).  
- **Maximum Transaction Date**: February 28, 2020 (suggesting possible truncation of recent data).  

---

## **3. Fraud Analysis**
- **Fraudulent Transactions**: 388,431 cases, primarily in the "Insufficient Balance" category (242,783 cases).  
- **Transaction Amounts**: Potential for fraud transactions to have higher mean amounts compared to non-fraudulent ones.  
- **Patterns**: Fraud may be concentrated in specific merchants, occur at certain hours (e.g., late nights), or involve specific card types.  

---

## **4. Transaction Amount Patterns**
- **Amount Column**: Stored as strings (`"$80.00"` format), requiring conversion to numeric for analysis.  
- **High Frequency at $80.00**: Suggests potential rounding or system-imposed transaction limits.  

---

## **5. Merchant Behavior**
- **Merchant Cities vs. Merchants**: 13,429 unique cities and 98,953 unique merchants, indicating some merchants operate in multiple cities.  
- **Top Merchant City**: "ONLINE," reflecting the dominance of e-commerce transactions.  
- **Most Common Merchant State**: "CA" (California), raising questions about fraud concentration in specific states.  

---

## **6. MCC (Merchant Category Code) & Zip Code**
- **Missing ZIP Codes**: 2.15M missing values, potentially linked to online transactions.  
- **MCC Values**: Some negative MCC codes, indicating possible data corruption or errors.  

---

## **7. Errors & Transaction Declines**
- **Error Types**: 23 unique error types, with "Insufficient Balance" being the most common (242,783 occurrences).  
- **Fraud Connection**: Fraudulent transactions often involve balance-related declines.  
- **Patterns**: Potential for repeated failed attempts on the same card before a successful fraud transaction.  

---

This descriptive analysis highlights key characteristics, patterns, and potential issues in the transactional dataset. Let me know if you'd like to explore any specific aspect further!