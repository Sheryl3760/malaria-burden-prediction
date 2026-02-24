# Predicting Malaria Burden in Africa
### A Comparative Study of Traditional Machine Learning and Deep Learning Approaches

**Author:** Sherry Otieno  
**Institution:** African Leadership University  
**Programme:** Machine Learning and Mobile Application Development  
**Date:** February 2026

---

## Overview

Malaria remains one of the most devastating infectious diseases in Sub-Saharan Africa, accounting for approximately 94% of global cases and 95% of malaria-related deaths in 2023 (WHO, 2024). This project investigates whether a country's health infrastructure indicators — access to clean water, sanitation services, insecticide-treated bed nets, and population distribution — can be used to accurately predict its malaria burden category.

The task is framed as a multi-class classification problem with three target labels:

- Low — incidence below 100 per 1,000 population at risk
- Medium — incidence between 100 and 300
- High — incidence above 300

Nine systematically designed experiments compare traditional machine learning (Logistic Regression, Random Forest, XGBoost) against deep learning (TensorFlow Sequential API, Functional API, tf.data pipeline) on the same dataset.

---

## Repository Contents

```
malaria-burden-prediction/
│
├── malaria_notebook.ipynb       # Main Jupyter notebook with all 9 experiments
└── README.md                    # This file
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | World Bank / WHO health indicators |
| Countries | 54 African countries |
| Time span | 2007 to 2017 |
| Original rows | 594 |
| Rows after cleaning | 552 |
| Features after engineering | 25 |
| Target classes | Low / Medium / High malaria burden |

Key features include access to basic drinking water services (overall, urban, and rural), access to basic sanitation services, use of insecticide-treated bed nets among children under five, antimalarial drug receipt among children with fever, intermittent preventive treatment coverage in pregnancy, and geographic coordinates.

Note on data quality: Several columns have 70 to 85 percent missing values, reflecting real-world limitations in African health surveillance data. These are handled through per-country mean imputation with a global mean fallback.

---

## Experiments Summary

| No. | Experiment | Type | Accuracy | ROC-AUC |
|-----|-----------|------|----------|---------|
| 1 | Logistic Regression (L2) | Traditional ML | 0.9009 | 0.9890 |
| 2 | Logistic Regression (L1) | Traditional ML | 0.9099 | 0.9909 |
| 3 | Random Forest (Default) | Traditional ML | 0.9279 | 0.9946 |
| 4 | Random Forest (Tuned) | Traditional ML | 0.9189 | 0.9929 |
| 5 | XGBoost | Traditional ML | 0.9369 | 0.9949 |
| 6 | Simple Sequential DL | Deep Learning | 0.9369 | 0.9935 |
| 7 | Sequential + Dropout + BatchNorm | Deep Learning | 0.9189 | 0.9909 |
| 8 | Functional API | Deep Learning | 0.8919 | 0.9899 |
| 9 | tf.data + SGD + ReduceLROnPlateau | Deep Learning | 0.9099 | 0.9911 |

Best models: XGBoost and Simple Sequential DL tied at 93.69% accuracy.

---

## Notebook Structure

The notebook is organised into 10 clearly labelled sections:

| Section | Description |
|---------|-------------|
| 1. Imports and Setup | All library imports and random seed configuration |
| 2. Data Loading | Load dataset, inspect shape, data types, statistics, and missing values |
| 3. Exploratory Data Analysis | Distribution plots, top countries by burden, yearly trends, correlation heatmap |
| 4. Preprocessing | Drop redundant columns, per-country mean imputation, verify no nulls remain |
| 5. Feature Engineering | Target variable binning, composite features, encoding, train/val/test split, standardisation |
| 6. Helper Functions | Modular reusable functions for evaluation, confusion matrices, ROC curves, and learning curves |
| 7. Experiments 1 to 9 | All nine experiments with rationale before and interpretation after each |
| 8. Results Summary | Comparative results table and bar charts across all experiments |
| 9. Conclusions | Key findings, bias-variance analysis, dataset limitations, and future work |

---

## Key Findings

- XGBoost and a simple sequential neural network tied at the top with 93.69% accuracy, demonstrating that gradient boosting is highly competitive with deep learning on small structured datasets.
- Deep learning did not clearly outperform traditional ML. With only 352 training rows, there is insufficient data for deep learning to leverage its additional capacity.
- The Medium burden class was the hardest to classify across all nine experiments, with precision ranging from 0.79 to 0.88, because intermediate-burden countries share infrastructure characteristics with both Low and High burden neighbours in the feature space.
- Hyperparameter tuning of Random Forest reduced accuracy compared to default settings, illustrating the bias-variance trade-off in a data-limited setting.
- Regularisation in deep learning (Dropout and Batch Normalisation) reduced overfitting but also reduced accuracy slightly, a direct demonstration of the bias-variance trade-off.
- Country identity (target-encoded as mean malaria incidence) and Year were the top features in the Random Forest importance analysis.

---

## Preprocessing Pipeline

```
Raw Dataset (594 rows, 27 columns)
        |
        v
Drop redundant columns (geometry, Country Code)
        |
        v
Per-country mean imputation -> global mean fallback
        |
        v
Create target variable (bin incidence into Low / Medium / High)
        |
        v
Feature engineering (Water_Sanitation_Score, Urban_Rural_Water_Gap)
        |
        v
Target encoding for Country Name
        |
        v
Label encoding for target variable
        |
        v
Stratified train / val / test split (64% / 16% / 20%)
        |
        v
StandardScaler (fit on train only, transform val and test)
        |
        v
Ready for modelling (552 rows, 25 features)
```

---

## How to Run

### Option 1: Kaggle (Recommended)

The notebook was developed and tested on Kaggle with a Tesla P100 GPU.

1. Go to kaggle.com and create a free account
2. Create a new notebook
3. Upload the dataset file via Add Data
4. Upload the notebook or paste the code
5. Update the dataset path in the data loading cell to match your Kaggle input path
6. Set Settings > Accelerator > GPU for faster deep learning training
7. Click Run All

### Option 2: Google Colab

1. Open colab.research.google.com
2. Upload the notebook via File > Upload notebook
3. Upload the dataset to Colab's file system or link it from Google Drive
4. Update the dataset path in the data loading cell
5. Click Runtime > Run all

### Option 3: Local Environment

```bash
git clone https://github.com/YOUR_USERNAME/malaria-burden-prediction.git
cd malaria-burden-prediction
pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow jupyter
jupyter notebook malaria_notebook.ipynb
```

---

## Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
tensorflow >= 2.10
jupyter
```

All libraries are pre-installed on Kaggle. For other environments, install via pip followed by the library names listed above.

---

## Reproducibility

- Random seed 42 is set at the start of the notebook for both NumPy and TensorFlow
- All scikit-learn models use random_state=42
- Train/val/test splits use random_state=42 with stratified sampling
- The notebook runs top to bottom without errors and requires no manual cell reordering
- All experiments were run on Kaggle using a Tesla P100 GPU

Note: Deep learning results (Experiments 6 to 9) may vary slightly across different hardware environments due to GPU-specific floating point behaviour, even with seeds set.

---

## Limitations

- Several columns have 70 to 85 percent missing values, requiring extensive imputation that may introduce systematic bias
- The dataset covers 2007 to 2017 only and may not reflect current malaria dynamics
- With only 352 training observations, deep learning models cannot fully leverage their capacity advantage over traditional ML
- Binning continuous incidence into three classes introduces boundary sensitivity near the thresholds of 100 and 300

---

## Future Work

- Incorporate climate variables such as rainfall, temperature, and humidity alongside infrastructure indicators
- Extend the dataset to include more recent years from 2018 to 2024
- Apply multiple imputation by chained equations (MICE) to handle missing values more robustly
- Explore LSTM or Transformer architectures that leverage the 11-year temporal structure of the data
- Apply systematic hyperparameter optimisation using tools such as Optuna
- Investigate whether oversampling specifically for the Medium burden class improves precision

---

## Acknowledgements

- Dataset sourced from the World Bank and WHO health indicator databases
- Experiments conducted on Kaggle with a Tesla P100 GPU
- WHO World Malaria Report 2024 for epidemiological context and statistics

---

## License

This project is submitted as academic coursework at African Leadership University. The dataset is publicly available from the World Bank and WHO. Code is available for educational and research use.
