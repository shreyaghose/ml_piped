# Understanding Data Splitting Errors in Data Pipelines

When constructing data pipelines for machine learning models, it's crucial to handle the splitting of data properly. Splitting the data at the wrong stage in the pipeline can lead to data leakage and invalid model evaluations. In this guide, we will explain the differences between two example pipelines and provide practical insights into the data splitting error.

## The Problem: Data Splitting Errors

The data splitting error occurs when the data is split into training and test sets at an incorrect stage of the pipeline. This typically happens when the data is split after some preprocessing steps have already been applied to the entire dataset, rather than splitting the raw data first. This can lead to data leakage, where information from the test set inadvertently influences the training process, resulting in overly optimistic performance estimates.

## Practical Advice

1. **Splitting Early**: It is recommended to split the raw dataset into training and test sets as soon as possible, ideally immediately after loading the raw data.
2. **Separate Preprocessing**: The preprocessing steps should be applied separately to the training and test sets to avoid any information leakage.
3. **Pipeline Construction**: The correct order of operations - data extraction, data splitting, and then performing data transformation, model training, and evaluation, should be ensured in a pipeline.

## References
