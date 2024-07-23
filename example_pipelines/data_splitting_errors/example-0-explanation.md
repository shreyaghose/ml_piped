# Understanding Data Splitting Error in Data Pipelines

When constructing data pipelines for machine learning models, it's crucial to handle the splitting of data properly. Splitting the data at the wrong stage in the pipeline can lead to data leakage and invalid model evaluations. In this guide, we will explain the differences between two example pipelines and provide practical insights into the data splitting error.

## The Problem: Data Splitting Error

The data splitting error occurs when the data is split into training and test sets at an incorrect stage of the pipeline. This typically happens when the data is split after some preprocessing steps have already been applied to the entire dataset, rather than splitting the raw data first. This can lead to data leakage, where information from the test set inadvertently influences the training process, resulting in overly optimistic performance estimates.

## Understanding the Example Pipelines

### Incorrect Pipeline (example-0.py)

An example incorrect pipeline is located [here](./example-0.py).

### Correct Pipeline (example-0-fixed.py)

The fixed correct pipeline is located [here](./example-0-fixed.py).

### Pipeline Details
1. **Aim**: To classify offenders into those that are likely to commit another crime (score_text = High) versus those that are not (score_text = Low)
2. **Dataset Used**: [compas-scores-two-years.csv](datasets/compas-scores-two-years.csv)
3. **Labels/Ground Truth**: The `score_text' column present in the dataset was used as row labels
4. **Label Binarization**: The pipeline binarizes 'High' score_text labels to 0 and 'Low' labels to 1
5. **Data Splitting**: Both pipelines split the data into train and test sets at a ratio of 80:20
6. **Train and Test Sizes**:
   - **example-0.py**: Shape of training data: (4937, 3); Shape of testing data: (1235, 3)
   - **example-0-fixed.py**: Shape of training data: (5771, 53); Shape of testing data: (1443, 53)
7. **Classifier Used**: Logistic Regression Classifier
8. **Classification Accuracy**:
   - **example-0.py**: 0.8186
   - **example-0-fixed.py**: 0.7969

## Key Differences

1. **Data Splitting Stage**:
   - **example-0.py**: Data is split after preprocessing steps are applied to the entire dataset.
   - **example-0-fixed.py**: Data is split before any preprocessing steps are applied.

2. **Risk of Data Leakage**:
   - **example-0.py**: Preprocessing on the entire dataset can lead to data leakage, as information from the test set influences the training set.
   - **example-0-fixed.py**: Preprocessing is done separately on training and test sets, preventing data leakage.

3. **Train and Test Set Sizes**:
    - **example-0.py**: The smaller number of features (3) suggests that the incorrect preprocessing reduced the dataset's dimensionality, possibly losing important information.
   - **example-0-fixed.py**: The larger number of features (53) shows that the fixed pipeline retained more information before splitting, leading to a more comprehensive feature set for training and testing.

3. **Model Evaluation**:
   - **example-0.py**: The model evaluation is not trustworthy (please see output comparison below) due to potential data leakage.
   - **example-0-fixed.py**: The model evaluation is reliable and reflects the true performance. 
   
## Comparison of Outputs

When running the pipelines with consistent randomization and preparation steps, the outputs can be compared. The pipelines were both randomized with 'random_seed = 42'.

### Classification Report for example-0.py:
The model shows high accuracy primarily due to excellent performance on class 1 (score_text = Low), but it completely fails to identify instances of class 0 (score_text = High), indicating possible class imbalance issues and potential data leakage.

|                | precision | recall | f1-score | support |
|----------------|-----------|--------|----------|---------|
| 0              | 0.00      | 0.00   | 0.00     | 224     |
| 1              | 0.82      | 1.00   | 0.90     | 1011    |
| **accuracy**   |           |        | 0.82     | 1235    |
| **macro avg**  | 0.41      | 0.50   | 0.45     | 1235    |
| **weighted avg** | 0.67   | 0.82   | 0.74     | 1235    |


### Classification Report for example-0-fixed.py :
The model performs well on class 1 (score_text = Low) with high precision and recall but struggles with class 0 (score_text = High), which may suggest an inherent difficulty in predicting this class but without the data leakage seen in the incorrect pipeline (see 'Class Imbalance' below). 

|                | precision | recall | f1-score | support |
|----------------|-----------|--------|----------|---------|
| 0              | 0.00      | 0.00   | 0.00     | 293     |
| 1              | 0.80      | 1.00   | 0.89     | 1150    |
| **accuracy**   |           |        | 0.80     | 1443    |
| **macro avg**  | 0.40      | 0.50   | 0.44     | 1443    |
| **weighted avg** | 0.64   | 0.80   | 0.71     | 1443    |


### Output Interpretation

The differences in the outputs between the incorrect and correct pipelines highlight the impact of the data splitting error:

1. **Model Performance**:
   - **example-0.py**: The accuracy is slightly higher, but the precision and recall for class 0 are poor, indicating potential data leakage from preparation of the data without splitting. The model is likely overfitting to the training data, leading to unrealistic performance estimates.
   - **example-0-fixed.py**: The accuracy is slightly lower, but the precision and recall metrics are more balanced and realistic. This indicates a more reliable evaluation of the model's performance.

2. **Generalization**:
   - **example-0.py**: Due to data leakage, the model may not generalize well to unseen data, as the evaluation metrics are inflated.
   - **example-0-fixed.py**: The model is more likely to generalize well to new data, as the evaluation reflects true performance without any leakage.

3. **Class Imbalance**:
    - **example-0.py** and **example-0-fixed.py**: The classification reports of both pipelines indicate that the frequency of the class score_text = High is much less compared to the class score_text = Low in the raw dataset, and thus by extension in the training and testing datasets. This can be confirmed with a distribution check and indicates class imbalance.

## Practical Advice

1. **Splitting Early**: It is recommended to split the raw dataset into training and test sets as soon as possible, ideally immediately after loading the raw data.
2. **Separate Preprocessing**: The preprocessing steps should be applied separately to the training and test sets to avoid any information leakage.
3. **Pipeline Construction**: The correct order of operations - data extraction, data splitting, and then performing data transformation, model training, and evaluation, should be ensured in a pipeline.
