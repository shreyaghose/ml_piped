# Understanding Data Leakage in Data Pipelines
When constructing data pipelines for machine learning models, it's crucial to prevent data leakage. Data leakage occurs when information from outside the training dataset is used to create the model, leading to overly optimistic performance estimates and poor generalization to new data. In this guide, we will explain the impact of data leakage, how it occurs, and practical steps to avoid it.

## The Problem: Data Leakage Errors
Data leakage happens when information from the test set (or future data in time-series models) is used during the training process. This can happen inadvertently through various stages of data preprocessing or feature engineering. Performing imputation, scaling, encoding, or other transformations on the entire dataset before splitting into training and test sets can cause leakage. Leakage results in models that perform well on seen data but fail to generalize to unseen data, providing misleading performance metrics.

## Practical Advice
1. **Splitting Early**: It is recommended to split the raw dataset into training and test sets as soon as possible, ideally immediately after loading the raw data to ensure that no information from the test set contaminates the training process.

2. **Separate Preprocessing**: The preprocessing steps should be applied separately to the training and test sets to avoid any information leakage.

3. **Pipeline Construction**: Data pipelines that encapsulate preprocessing and model training steps should be constructed ensuring they are fit on the training data and then applied to the test data to maintain the separation of training and test data throughout the preprocessing and model training stages.

## References
1. Yang, C., Brower-Sinning, R.A., Lewis, G. and Kästner, C., 2022, October. Data leakage in notebooks: Static detection and better processes. In Proceedings of the 37th IEEE/ACM International Conference on Automated Software Engineering (pp. 1-12).
2. Lones, M.A., 2021. How to avoid machine learning pitfalls: a guide for academic researchers. arXiv preprint arXiv:2108.02497.
3. An empirical study of pattern leakage impact during data preprocessing on machine learning-based intrusion detection models reliability
4. Kaufman, S., Rosset, S., Perlich, C. and Stitelman, O., 2012. Leakage in data mining: Formulation, detection, and avoidance. ACM Transactions on Knowledge Discovery from Data (TKDD), 6(4), pp.1-21.
5. Hewamalage, H., Ackermann, K. and Bergmeir, C., 2023. Forecast evaluation for data scientists: common pitfalls and best practices. Data Mining and Knowledge Discovery, 37(2), pp.788-832.
6. Kapoor, S. and Narayanan, A., 2022. Leakage and the reproducibility crisis in ML-based science. arXiv preprint arXiv:2207.07048.