import os
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, label_binarize, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "compas_scores", "compas-scores-two-years.csv")
raw_data = pd.read_csv(raw_data_file)

# Data preparation steps
# Data Extraction
raw_data = raw_data[
    ['sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count', 'days_b_screening_arrest',
     'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]

# Data Replacement
raw_data = raw_data.replace('Medium', "Low")

# Checking for representational bias in the training data
def check_representational_bias(data, protected_attribute):
    counter = Counter(data[protected_attribute])
    print(f"Distribution of {protected_attribute} in the data: {counter}")

# Data Splitting
train_data, test_data, train_labels, test_labels = train_test_split(raw_data, raw_data['score_text'], test_size=0.2, random_state=42)
print("Shape of training data:", train_data.shape)
print("Shape of testing data:", test_data.shape)

# Imputing missing values and encode categorical variables
categorical_features = ['sex', 'race', 'c_charge_degree']
numeric_features = ['age', 'priors_count', 'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid']

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Fitting and transform the training data
train_data_encoded = preprocessor.fit_transform(train_data)

# Transforming the test data
test_data_encoded = preprocessor.transform(test_data)

# Binarizing labels
train_labels = label_binarize(train_labels, classes=['High', 'Low'])
test_labels = label_binarize(test_labels, classes=['High', 'Low'])

# Checking for class distribution before SMOTE
print("Class distribution before SMOTE:", Counter(train_labels.flatten()))

# Mitigate bias using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(sampling_strategy='auto', random_state=42)
train_data_smote, train_labels_smote = smote.fit_resample(train_data_encoded, train_labels.ravel())

# Checking the distribution after SMOTE
print("Distribution of classes after SMOTE:", Counter(train_labels_smote))

# Model Evaluation
pipeline = Pipeline([('classifier', LogisticRegression(max_iter=1000))])

pipeline.fit(train_data_smote, train_labels_smote)
print("Accuracy", pipeline.score(test_data_encoded, test_labels.ravel()))

# Classification Report
predictions = pipeline.predict(test_data_encoded)
print(classification_report(test_labels, predictions, zero_division=0))

# Evaluating model fairness across protected attribute groups
def evaluate_fairness(predictions, test_data, protected_attribute):
    # Creating a DataFrame with predictions and the protected attribute
    fairness_df = test_data.copy()
    fairness_df['predictions'] = predictions
    fairness_df['actual'] = test_labels.ravel()  # Add actual labels for comparison
    
    # Resetting the index of fairness_df to avoid alignment issues
    fairness_df = fairness_df.reset_index(drop=True)
    
    group_stats = fairness_df.groupby(protected_attribute).agg(
        accuracy=('predictions', lambda x: (x == fairness_df.loc[x.index, 'actual']).mean()),  # Ensuring alignment
        count=('predictions', 'count')
    )
    
    print("Model Fairness Metrics:")
    print(group_stats)

# Evaluating fairness based on 'race'
evaluate_fairness(predictions, test_data, 'race')