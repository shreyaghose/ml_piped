import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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

raw_data = raw_data[
    ['sex', 'dob', 'age', 'c_charge_degree', 'race', 'score_text', 'priors_count', 'days_b_screening_arrest',
     'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]

# Filling missing values
for column in raw_data.columns:
    if raw_data[column].dtype == 'object':
        raw_data[column] = raw_data[column].fillna(raw_data[column].mode()[0])
    else:
        raw_data[column] = raw_data[column].fillna(raw_data[column].mean())

X = raw_data.drop(columns=['score_text'])
y = raw_data['score_text']

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identifying categorical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# Converting categorical columns to numeric using one-hot encoding on training data
X_train = pd.get_dummies(X_train, columns=categorical_cols)

# Ensuring the same columns are present in the test set
X_test = pd.get_dummies(X_test, columns=categorical_cols)

# Aligning the test set to the training set
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Training a simple classifier with arbitrary hyper-parameters
clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Prediction and evaluation
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification report: {classification_report(y_test, y_pred)}")