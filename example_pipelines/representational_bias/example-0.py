# All imports
import os
import sys
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, label_binarize
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

# Data preparation steps. Note: The data is not checked for representational bias
# Data Extraction
raw_data = raw_data[
    ['sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count', 'days_b_screening_arrest',
     'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]

# Data Replacement
raw_data = raw_data.replace('Medium', "Low")

# Data Preparation Pipeline (Imputation, Encoding, Discretization)
impute1_and_onehot = Pipeline([('imputer1', SimpleImputer(strategy='most_frequent')),
                               ('onehot', OneHotEncoder(handle_unknown='ignore'))])
impute2_and_bin = Pipeline([('imputer2', SimpleImputer(strategy='mean')),
                            ('discretizer', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'))])

featurizer = ColumnTransformer(transformers=[
    ('impute1_and_onehot', impute1_and_onehot, ['is_recid']),
    ('impute2_and_bin', impute2_and_bin, ['age'])
])

# Dropping labels for the data split
data_without_labels = raw_data.copy(deep=True)
data_without_labels = data_without_labels.drop(['score_text'], axis=1)

# Data fitted to featurizer without split
prepared_data = featurizer.fit_transform(data_without_labels)

# Data Splitting
train_data, test_data, train_labels, test_labels = train_test_split(prepared_data, raw_data['score_text'], test_size=0.2, random_state=42)
print("Shape of training data:", train_data.shape)
print("Shape of testing data:", test_data.shape)

# Binarizing labels
train_labels = label_binarize(train_labels, classes=['High', 'Low'])
test_labels = label_binarize(test_labels, classes=['High', 'Low'])

# Model Evaluation
pipeline = Pipeline([('classifier', LogisticRegression())])

pipeline.fit(train_data, train_labels.ravel())
print("Accuracy", pipeline.score(test_data, test_labels.ravel()))

# Classification Report
print(classification_report(test_labels, pipeline.predict(test_data), zero_division=0))