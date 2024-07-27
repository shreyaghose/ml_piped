# All imports
import os
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "compas_scores", "compas-scores-two-years.csv")
raw_data = pd.read_csv(raw_data_file)

# Data Splitting before Data Preparation
train_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=42)

# Data Extraction
columns_to_use = ['sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count', 'days_b_screening_arrest',
                  'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']

train_data = train_data[columns_to_use]
test_data = test_data[columns_to_use]

print("Shape of training data:", train_data.shape)
print("Shape of testing data:", test_data.shape)

# Data Replacement
train_data = train_data.replace('Medium', "Low")
test_data = test_data.replace('Medium', "Low")

# Binarizing labels
train_labels = label_binarize(train_data['score_text'], classes=['High', 'Low']).ravel()
test_labels = label_binarize(test_data['score_text'], classes=['High', 'Low']).ravel()

# Drop the 'score_text' column from features
train_data = train_data.drop(columns=['score_text'])
test_data = test_data.drop(columns=['score_text'])

# Resetting indices to avoid alignment issues
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

# Data Preparation Pipeline (Imputation, Encoding, Discretization). Note: Applied to unsplit, raw data
impute1_and_onehot = Pipeline([('imputer1', SimpleImputer(strategy='most_frequent')),
                               ('onehot', OneHotEncoder(handle_unknown='ignore'))])
impute2_and_bin = Pipeline([('imputer2', SimpleImputer(strategy='mean')),
                            ('discretizer', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'))])

featurizer = ColumnTransformer(transformers=[
    ('impute1_and_onehot', impute1_and_onehot, ['is_recid']),
    ('impute2_and_bin', impute2_and_bin, ['age'])
])

# Complete pipeline with preprocessing and classification
pipeline = Pipeline(steps=[('featurizer', featurizer),
    ('classifier', LogisticRegression())
])

# Training the pipeline
pipeline.fit(train_data, train_labels)

# Model Evaluation
print("Model score:", pipeline.score(test_data, test_labels))

# Classification Report
print(classification_report(test_labels, pipeline.predict(test_data), zero_division=0))

# # Performance Evaluation for Slices
# slices = test_data.groupby(['race', 'sex'])

# for slice_name, slice_df in slices:
#     slice_indices = slice_df.index
#     slice_y_true = test_labels[slice_indices]
#     slice_y_pred = pipeline.predict(slice_df)
#     print(f"Performance for slice {slice_name}:")
#     print(classification_report(slice_y_true, slice_y_pred, zero_division=0))