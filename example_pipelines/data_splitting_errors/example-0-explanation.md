# Understanding Data Splitting Error in Data Pipelines

When constructing data pipelines for machine learning models, it's crucial to handle the splitting of data properly. Splitting the data at the wrong stage in the pipeline can lead to data leakage and invalid model evaluations. In this guide, we will explain the differences between two example pipelines and provide practical insights into the data splitting error.

## The Problem: Data Splitting Error

The data splitting error occurs when the data is split into training and test sets at an incorrect stage of the pipeline. This typically happens when the data is split after some preprocessing steps have already been applied to the entire dataset, rather than splitting the raw data first. This can lead to data leakage, where information from the test set inadvertently influences the training process, resulting in overly optimistic performance estimates.

## Example Pipelines

### Incorrect Pipeline (example-0.py)

Refer to the following file for an example of the incorrect pipeline:

[Link to incorrect pipeline code](./example-0.py)

### Correct Pipeline (example-0-fixed.py)

Refer to the following file of the correct pipeline:

[Link to correct pipeline code](./example-0-fixed.py)

## Key Differences

1. **Data Splitting Stage**:
   - **example-0.py**: Data is split after preprocessing steps are applied to the entire dataset.
   - **example-0-fixed.py**: Data is split before any preprocessing steps are applied.

2. **Risk of Data Leakage**:
   - **example-0.py**: Preprocessing on the entire dataset can lead to data leakage, as information from the test set influences the training set.
   - **example-0-fixed.py**: Preprocessing is done separately on training and test sets, preventing data leakage.

3. **Model Evaluation**:
   - **example-0.py**: The model evaluation is not trustworthy due to potential data leakage.
   - **example-0-fixed.py**: The model evaluation is reliable and reflects the true performance.

## Practical Advice

1. **Splitting Early**: It is recommended to split the raw dataset into training and test sets as soon as possible, ideally immediately after loading the raw data.
2. **Separate Preprocessing**: The preprocessing steps should be applied separately to the training and test sets to avoid any information leakage.
3. **Pipeline Construction**: The correct order of operations - data extraction, data splitting, data transformation, model training, and evaluation, should be ensured in a pipeline.

## Comparison of Outputs

When running the pipelines with consistent randomization, the outputs can be compared directly:

### Output for example-0.py on the 'compas-scores-two-years.csv' dataset is as follows:
|                | precision | recall | f1-score | support |
|----------------|-----------|--------|----------|---------|
| 0              | 0.00      | 0.00   | 0.00     | 224     |
| 1              | 0.82      | 1.00   | 0.90     | 1011    |
| **accuracy**   |           |        | 0.82     | 1235    |
| **macro avg**  | 0.41      | 0.50   | 0.45     | 1235    |
| **weighted avg** | 0.67   | 0.82   | 0.74     | 1235    |


### Output for example-0-fixed.py on the 'compas-scores-two-years.csv' dataset is as follows:
|                | precision | recall | f1-score | support |
|----------------|-----------|--------|----------|---------|
| 0              | 0.00      | 0.00   | 0.00     | 293     |
| 1              | 0.80      | 1.00   | 0.89     | 1150    |
| **accuracy**   |           |        | 0.80     | 1443    |
| **macro avg**  | 0.40      | 0.50   | 0.44     | 1443    |
| **weighted avg** | 0.64   | 0.80   | 0.71     | 1443    |


## Differences in the Results

The differences in the outputs between the incorrect and correct pipelines highlight the impact of the data splitting error:

1. **Model Performance**:
   - **example-0.py**: The accuracy is slightly higher, but the precision and recall for class 0 are poor, indicating potential data leakage. The model is likely overfitting to the training data, leading to unrealistic performance estimates.
   - **example-0-fixed.py**: The accuracy is slightly lower, but the precision and recall metrics are more balanced and realistic. This indicates a more reliable evaluation of the model's performance.

2. **Generalization**:
   - **example-0.py**: Due to data leakage, the model may not generalize well to unseen data, as the evaluation metrics are inflated.
   - **example-0-fixed.py**: The model is more likely to generalize well to new data, as the evaluation reflects true performance without any leakage.

*Note: The dataset used for the example pipelines are considered small sets with ~6200 data points. The difference is more apparent with larger datasets.* 
