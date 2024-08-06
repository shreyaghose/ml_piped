import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import numpy as np

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "diabetes_indicator", "binary_health_indicators.csv")
data = pd.read_csv(raw_data_file)

# Splitting dataset into features and target
X = data.drop(columns=['Diabetes_binary'])
y = data['Diabetes_binary']

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Scale features for better performance
    ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="mean")),
    ('classifier', LogisticRegression(random_state=42))
])

# Training the pipeline
pipeline.fit(X_train, y_train)

# Predicting and evaluation
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Performing cross-validation to check for robustness
cv_scores = cross_val_score(pipeline, X, y, cv=5)
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {np.mean(cv_scores)}")