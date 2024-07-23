# All imports
import os
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Getting the project root
project_root = Path.cwd()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "compas-scores-two-years.csv")
raw_data = pd.read_csv(raw_data_file)

# Data Splitting before Data Preparation
train_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=42)

# Data Extraction
train_data = train_data[
    ['sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count', 'days_b_screening_arrest',
     'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]

test_data = test_data[
    ['sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count', 'days_b_screening_arrest',
     'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]

# Data Filtering. Note: This is only performed for the train set
train_data = train_data[(train_data['days_b_screening_arrest'] <= 30) & (train_data['days_b_screening_arrest'] >= -30)]
train_data = train_data[train_data['is_recid'] != -1]
train_data = train_data[train_data['c_charge_degree'] != "O"]
train_data = train_data[train_data['score_text'] != 'N/A']

# Uncomment to test for accuracy after filtering test set
# test_data = test_data[(test_data['days_b_screening_arrest'] <= 30) & (test_data['days_b_screening_arrest'] >= -30)]
# test_data = test_data[test_data['is_recid'] != -1]
# test_data = test_data[test_data['c_charge_degree'] != "O"]
# test_data = test_data[test_data['score_text'] != 'N/A']

# Data Replacement
train_data = train_data.replace('Medium', "Low")
test_data = test_data.replace('Medium', "Low")

# Binarizing labels
train_labels = label_binarize(train_data['score_text'], classes=['High', 'Low'])
test_labels = label_binarize(test_data['score_text'], classes=['High', 'Low'])

# Data Preparation Pipeline (Imputation, Encoding, Discretization)
impute1_and_onehot = Pipeline([('imputer1', SimpleImputer(strategy='most_frequent')),
                               ('onehot', OneHotEncoder(handle_unknown='ignore'))])
impute2_and_bin = Pipeline([('imputer2', SimpleImputer(strategy='mean')),
                            ('discretizer', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'))])

# Note that the featurizer 
featurizer = ColumnTransformer(transformers=[
    ('impute1_and_onehot', impute1_and_onehot, ['is_recid']),
    ('impute2_and_bin', impute2_and_bin, ['age'])
])

train_data = featurizer.fit_transform(train_data)
test_data = featurizer.fit_transform(test_data)

print("Shape of training data:", train_data.shape)
print("Shape of testing data:", test_data.shape)

pipeline = Pipeline([('classifier', LogisticRegression())])

# Model Evaluation
pipeline.fit(train_data, train_labels.ravel())
print(pipeline.score(test_data, test_labels.ravel()))

# Classification Report
print(classification_report(test_labels, pipeline.predict(test_data), zero_division=0))