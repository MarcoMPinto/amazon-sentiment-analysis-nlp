# 📦 Amazon Sentiment Analysis: High-Recall NLP Pipeline - amazon-sentiment-analysis-nlp

### **Project Overview**
This repository contains a complete End-to-End Machine Learning pipeline for Sentiment Analysis, using a dataset of over **194,000 Amazon reviews**. The project addresses the challenge of **Class Imbalance** (86% Positive vs. 14% Negative) by implementing specialized weighting and optimization techniques to ensure high sensitivity to dissatisfied customers.

---

## **Key Result: 93% Recall for Negative Sentiment**
In many real-world applications, missing a negative signal is more critical than a false alarm. This model was tuned to prioritize **Recall**, successfully identifying **93% of all negative reviews** in the test set.

---

## **Technical Stack**
* **Language:** Python 3.10+
* **Data Processing:** `Pandas`, `NumPy`
* **Machine Learning:** `Scikit-Learn` (Pipeline, GridSearchCV, LogisticRegression)
* **NLP Techniques:** TF-IDF Vectorization, N-grams (Unigrams & Bigrams), Custom Regex Cleaning.
* **Visualization:** `Seaborn`, `Matplotlib`

---

## **Project Architecture**
1.  **Data Integrity:** Automated removal of null values and duplicates based on User ID and Review Text.
2.  **Text Normalization:** Custom preprocessing pipeline (Lowercase, Accent removal, Special character filtering).
3.  **Exploratory Data Analysis (EDA):** Statistical distribution of ratings and word frequency analysis.
4.  **Feature Engineering:** Conversion of raw text into a 5,000-feature TF-IDF matrix.
5.  **Model Optimization:** Use of `class_weight='balanced'` and `GridSearchCV` to find the optimal regularization parameter ($C$).
6.  **Production Ready:** Consolidation of the entire process into a single `Scikit-Learn Pipeline` for easy deployment.

---

## **Final Performance Metrics**

| Metric | Positive Class (1) | Negative Class (0) |
| :--- | :--- | :--- |
| **Precision** | 99% | 63% |
| **Recall** | 91% | **93%** |
| **F1-Score** | 95% | 75% |
| **Overall Accuracy** | **91.37%** | |

---

## **Usage**
1. **Requirements:** Install the necessary libraries using `pip install pandas scikit-learn matplotlib seaborn`.
2. **Execution:** Run the Jupyter Notebook to see the training flow and statistical analysis.
3. **Inference:** The project includes a live input function to predict sentiment on new, raw text strings.

---
