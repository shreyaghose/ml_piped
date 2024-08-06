import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
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

X = raw_data.drop('score_text', axis=1)
y = raw_data['score_text']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identifying numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Creating a column transformer with iterative imputer for numeric features and one hot encoder for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', IterativeImputer(random_state=42), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Defining pipeline with preprocessing and classifier
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Nested cross-validation for hyperparameter tuning and evaluation
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')

print("Cross-validation scores:", scores)

# Fitting on the full training data and evaluate on the test data
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("Test accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:", classification_report(y_test, y_pred))