# Predicting Malaria Burden in Africa
### A Comparative Study of Traditional Machine Learning and Deep Learning Approaches

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

## Key Findings

- XGBoost and a simple sequential neural network tied at the top with 93.69% accuracy, demonstrating that gradient boosting is highly competitive with deep learning on small structured datasets.
- Deep learning did not clearly outperform traditional ML. With only 352 training rows, there is insufficient data for deep learning to leverage its additional capacity.
- The Medium burden class was the hardest to classify across all nine experiments, with precision ranging from 0.79 to 0.88, because intermediate-burden countries share infrastructure characteristics with both Low and High burden neighbours in the feature space.
- Hyperparameter tuning of Random Forest reduced accuracy compared to default settings, illustrating the bias-variance trade-off in a data-limited setting.
- Regularisation in deep learning (Dropout and Batch Normalisation) reduced overfitting but also reduced accuracy slightly, a direct demonstration of the bias-variance trade-off.
- Country identity (target-encoded as mean malaria incidence) and Year were the top features in the Random Forest importance analysis.

---



## How to Run

### Option 1: Kaggle 

The notebook was developed and tested on Kaggle 

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

