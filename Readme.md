# FinTech Innovations — Credit Risk ML Pipeline

> **Building a Data-Driven Credit Risk Pipeline for Modern Banking**

![FinTech Loan Approval ML Pipeline](https://images.unsplash.com/photo-1560472354-b33ff0c44a43?w=1200&q=80)

---

## Overview

FinTech Innovations processes thousands of loan applications monthly through a manual review system that is inconsistent, slow, and prone to human bias. This project develops a **supervised machine learning classification pipeline**, trained on **20,000 historical loan applications**, to automate and standardize the approval decision process.

The final model achieves strong discrimination between creditworthy and high-risk applicants, evaluated primarily through **Precision-Recall AUC** and a custom **Business Cost Score** that directly quantifies the asymmetric cost structure of $8,000 (false negative) vs. $50,000 (false positive). Deploying this model as a decision-support layer for loan officers is projected to materially reduce default losses while preserving approval rates for creditworthy applicants.

---

## Table of Contents

1. [Business Understanding](#1-business-understanding)
2. [Dataset](#2-dataset)
3. [Methodology](#3-methodology)
4. [Results](#4-results)
5. [Feature Importance](#5-feature-importance)
6. [Business Cost Analysis](#6-business-cost-analysis)
7. [Recommendations](#7-recommendations)
8. [Limitations & Future Work](#8-limitations--future-work)
9. [Project Structure](#9-project-structure)
10. [Setup & Usage](#10-setup--usage)

---

## 1. Business Understanding

### The Problem

FinTech Innovations' manual loan review process has three critical failure modes:

| Failure Mode | Description |
|---|---|
| **Inconsistency** | Two officers reviewing the same application may reach different decisions, outcomes depend on *who* reviews the file, not just creditworthiness |
| **Speed** | Manual review creates processing bottlenecks that delay responses, damaging customer experience and competitive positioning |
| **Blind spots** | Human reviewers may overlook complex multivariate signals (e.g., moderate credit score + exceptional payment history + low utilization = low-risk applicant) |

### Cost of Model Errors

| Error Type | Description | Cost per Occurrence |
|---|---|---|
| **False Positive (FP)** | Approve a loan that defaults | **$50,000** (principal + collection costs) |
| **False Negative (FN)** | Deny a loan to a creditworthy applicant | **$8,000** (lost profit on a good loan) |

The **6.25:1 cost ratio** means approving one bad loan costs as much as wrongly denying 6.25 good ones. This drives a **precision-oriented** modeling strategy.

 A naive "deny everyone" classifier achieves 76.1% accuracy but generates **$38.24M in missed revenue** ($8,000 × 4,780 denied creditworthy applicants).

### Key Stakeholders

| Stakeholder | Primary Need |
|---|---|
| **Loan Officers** | A reliable risk signal to anchor judgment on borderline cases |
| **Risk Management / Leadership** | Minimized default losses, regulatory compliance, and audit trails |
| **Applicants** | Fast, fair, and consistent decisions |
| **Regulators** | Explainable, non-discriminatory decision logic |

### Success Criteria

| Metric | Baseline | Target | Achieved |
|---|---|---|---|
| Accuracy | 76.1% | > 82% | **96%** |
| PR-AUC | ~0.24 (random) | ≥ 0.75 | **0.9767** |
| Recall | 0% | ≥ 0.65 | **0.9048** |
| Business Cost Score | $38.24M | Minimize | **$4.13M** |

---

## 2. Dataset

- **Source:** `financial_loan_data.csv`
- **Size:** 20,000 loan applications × 35 features
- **Target:** `LoanApproved` (binary: 0 = Denied, 1 = Approved)
- **Class Distribution:** 76.1% Denied / 23.9% Approved (3.18:1 imbalance ratio)

### Feature Summary

| Category | Count | Examples |
|---|---|---|
| Numerical | 27 | `AnnualIncome`, `CreditScore`, `LoanAmount`, `NetWorth`, `TotalDebtToIncomeRatio` |
| Nominal Categorical | 5 | `EmploymentStatus`, `MaritalStatus`, `HomeOwnershipStatus`, `LoanPurpose`, `BankruptcyHistory` |
| Ordinal Categorical | 1 | `EducationLevel` (High School → Associate → Bachelor → Master) |

### Key Statistics

| Feature | Mean | Std | Min | Max |
|---|---|---|---|---|
| Age | 39.75 | 11.62 | 18 | 80 |
| CreditScore | 571.6 | 51.0 | 343 | 712 |
| LoanAmount | $24,883 | $13,427 | $3,674 | $184,732 |
| AnnualIncome | — | — | — | — |
| RiskScore | 50.77 | 7.78 | 28.8 | 84.0 |

### Data Quality Issues & Resolutions

| Issue | Resolution |
|---|---|
| `AnnualIncome` stored as string with `$` and commas | Stripped and cast to `float` |
| `MaritalStatus`: 1,331 missing (6.65%) | Imputed with mode in pipeline |
| `EducationLevel`: 901 missing (4.50%) | Imputed with mode; missing indicator added |
| `SavingsAccountBalance`: 572 missing (2.86%) | Imputed with median (right-skewed) |
| Class imbalance: 76.1% denied | `class_weight='balanced'`; evaluated with PR-AUC |
| Right-skewed financials (`NetWorth`, `TotalAssets`, etc.) | `StandardScaler` in pipeline |

**Note on `EducationLevel` missingness:** Approval rate drops to **0%** for records with missing `EducationLevel`, compared to 25.03% for records where it is present, a strong signal that warrants a missing indicator feature.

---

## 3. Methodology

### Framework

The project follows the **CRISP-DM** methodology: Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment.

### Train / Test Split

| Split | Rows | Denied | Approved |
|---|---|---|---|
| Train | 16,000 | 12,176 | 3,824 |
| Test | 4,000 | 3,044 | 956 |

Stratified split (80/20) to preserve class proportions.

### Preprocessing Pipeline

Built with scikit-learn `Pipeline` + `ColumnTransformer` to prevent data leakage:

- **Numerical:** `SimpleImputer(median)` → `StandardScaler`
- **Nominal Categorical:** `SimpleImputer(mode)` → `OneHotEncoder`
- **Ordinal Categorical:** `SimpleImputer(mode)` → `OrdinalEncoder` (ordered: High School < Associate < Bachelor < Master)

Transformed shape: **(16,000 × 46)** — no remaining NaN values.

### Models Evaluated

| Model | Rationale |
|---|---|
| **Logistic Regression** | Linear baseline; interpretable; required by regulators |
| **Random Forest** | Handles non-linearity; naturally resistant to multicollinearity |
| **Gradient Boosting** | Typically strongest on tabular data; good with imbalanced classes |

### Cross-Validation Results (5-Fold)

| Model | PR-AUC Mean | PR-AUC Std | Recall Mean | Recall Std |
|---|---|---|---|---|
| Logistic Regression | **0.9751** | 0.0031 | **0.9592** | 0.0070 |
| Gradient Boosting | 0.9573 | 0.0059 | 0.8496 | 0.0102 |
| Random Forest | 0.9301 | 0.0110 | 0.7772 | 0.0220 |

Logistic Regression was surprisingly competitive, suggesting the feature-approval relationship is largely linear. **Gradient Boosting** was selected as the final model for its stronger recall and better handling of non-linear interactions after tuning.

### Hyperparameter Tuning (Gradient Boosting)

Grid search over 8 candidates × 5 folds = 40 fits.

| n_estimators | max_depth | learning_rate | subsample | Mean PR-AUC | Rank |
|---|---|---|---|---|---|
| **200** | **5** | **0.10** | **0.8** | **0.9728** | **1** |
| 200 | 3 | 0.10 | 1.0 | 0.9708 | 2 |
| 100 | 5 | 0.10 | 1.0 | 0.9654 | 3 |
| 100 | 3 | 0.10 | 0.8 | 0.9582 | 4 |

**Best Parameters:** `n_estimators=200`, `max_depth=5`, `learning_rate=0.1`, `subsample=0.8`

---

## 4. Results

### Final Model Performance (Test Set — 4,000 rows)

| Metric | Value | Target |
|---|---|---|
| **PR-AUC** | **0.9767** | ≥ 0.75  |
| **Recall** | **0.9048** | ≥ 0.65 |
| **Precision** | **0.9271** | — |
| **F1 Score** | **0.9158** | — |
| **ROC-AUC** | **0.9916** | — |
| **Accuracy** | **96%** | > 82% |

### Classification Report

|  | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Denied | 0.97 | 0.98 | 0.97 | 3,044 |
| Approved | 0.93 | 0.90 | 0.92 | 956 |
| **Accuracy** | | | **0.96** | **4,000** |
| Macro Avg | 0.95 | 0.94 | 0.94 | 4,000 |
| Weighted Avg | 0.96 | 0.96 | 0.96 | 4,000 |

### Confusion Matrix

|  | Predicted: Denied | Predicted: Approved |
|---|---|---|
| **Actual: Denied** | 2,976 (TN) | 68 (FP) |
| **Actual: Approved** | 91 (FN) | 865 (TP) |

---

## 5. Feature Importance

Top 10 predictors from the tuned Gradient Boosting model:

| Rank | Feature | Importance |
|---|---|---|
| 1 | `TotalDebtToIncomeRatio` | 0.4611 |
| 2 | `InterestRate` | 0.1434 |
| 3 | `MonthlyIncome` | 0.1123 |
| 4 | `NetWorth` | 0.0491 |
| 5 | `LengthOfCreditHistory` | 0.0313 |
| 6 | `TotalAssets` | 0.0297 |
| 7 | `LoanDuration` | 0.0252 |
| 8 | `AnnualIncome` | 0.0248 |
| 9 | `LoanAmount` | 0.0198 |
| 10 | `CreditScore` | 0.0195 |

Top predictors are financial health indicators — `DebtToIncomeRatio`, `MonthlyIncome`, `NetWorth`, and `CreditScore` — which aligns with domain intuition and provides loan officers with explainable signals for borderline cases.

---

## 6. Business Cost Analysis

| Metric | Value |
|---|---|
| False Positives (bad loans approved) | 68 |
| False Negatives (good loans denied) | 91 |
| Cost of false positives | $3,400,000 |
| Cost of false negatives | $728,000 |
| **Total Business Cost** | **$4,128,000** |
| Baseline cost (deny all) | $7,648,000 |
| **Model saves vs. baseline** | **$3,520,000 (46% reduction)** |

False positives drive the majority of cost ($3.4M of $4.1M total), consistent with the 6.25:1 cost asymmetry. Further threshold tuning could reduce false positives at the cost of slightly higher false negatives.

---

## 7. Recommendations

1. **Deploy as decision support, not full automation.** Use the model to flag high-confidence approvals and denials, routing borderline cases to loan officers. This preserves human judgment where the cost of error is highest.

2. **Adjust the classification threshold.** The default 0.5 threshold can be raised to reduce the 68 false positives (each costing $50,000). Thresholds of 0.6–0.7 should be evaluated against the resulting increase in false negatives.

3. **Monitor for model drift.** Retrain quarterly as economic conditions shift. The 68 false positives in this test set represent $3.4M in potential defaults, even small drift in production warrants immediate attention.

---

## 8. Limitations & Future Work

### Limitations

- Training labels reflect past manual decisions, which may encode historical bias from loan officers. The model inherits any bias present in those decisions.
- `MaritalStatus` and `EducationLevel` carry fairness risk and should be reviewed by the compliance team before deployment.
- The dataset is imbalanced (76/24). Despite class weighting, performance on the approved class should be monitored continuously in production.

### Potential Improvements

- Apply **SHAP values** for per-application explainability to satisfy regulatory requirements.
- Explore **threshold optimization** using the $8,000/$50,000 cost function directly.
- Incorporate **alternative data sources** (utility payments, rent history) to better serve thin-file applicants who may be creditworthy but lack traditional credit signals.

---

## 9. Project Structure

```
loanAproval/
├── financial_loan_risk.ipynb   # Main analysis notebook (CRISP-DM)
├── financial_loan_data.csv     # Training dataset (20,000 records)
├── Loan.csv                    # Raw source data
└── Readme.md                   # This file
```

---

## 10. Setup & Usage

### Requirements

```
python >= 3.8
pandas
numpy
scikit-learn
matplotlib
seaborn
```

### Running the Notebook

```bash
# Clone or navigate to the project directory
cd loanAproval

# Launch Jupyter
jupyter notebook financial_loan_risk.ipynb
```

Run all cells in order. The notebook follows the CRISP-DM pipeline end-to-end:
1. Business Understanding
2. Data Understanding (EDA)
3. Data Preparation (preprocessing pipeline)
4. Modeling (cross-validation + hyperparameter tuning)
5. Evaluation (test set metrics + business cost analysis)
6. Conclusion & Recommendations

---

*Built with scikit-learn · Evaluated on 4,000 held-out test records · CRISP-DM methodology*
