# Understanding Data Leakage in Data Pipelines

When constructing data pipelines for machine learning models, it's crucial to prevent data leakage. Data leakage occurs when information from outside the training dataset is used to create the model, leading to overly optimistic performance estimates and poor generalization to new data. In this guide, we will explain the impact of data leakage, how it occurs, and practical steps to avoid it.

## The Problem: Data Leakage Errors

Data leakage happens when information from the test set (or future data in time-series models) is used during the training process. This can happen inadvertently through various stages of data preprocessing or feature engineering. Performing imputation, scaling, encoding, or other transformations on the entire dataset before splitting into training and test sets can cause leakage. Leakage results in models that perform well on seen data but fail to generalize to unseen data, providing misleading performance metrics.

## Practical Advice

1. **Splitting Early**: It is recommended to split the raw dataset into training and test sets as soon as possible, ideally immediately after loading the raw data to ensure that no information from the test set contaminates the training process.
2. **Separate Preprocessing**: The preprocessing steps should be applied separately to the training and test sets to avoid any information leakage.
3. **Pipeline Construction**: Data pipelines that encapsulate preprocessing and model training steps should be constructed ensuring they are fit on the training data and then applied to the test data to maintain the separation of training and test data throughout the preprocessing and model training stages.

## References
