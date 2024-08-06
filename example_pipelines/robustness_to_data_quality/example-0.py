import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
data = pd.read_csv(raw_data_file)

# Splitting the data
X = data.drop('salary', axis=1)
y = data['salary']

# Defining categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['number']).columns.tolist()

# Creating a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', 'passthrough', numerical_features)
    ],
    remainder='drop'  # Drop any other columns not specified
)

# Creating a pipeline that first transforms data and then fits the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier())  # Using Random Forest for sensitivity
])

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
pipeline.fit(X_train, y_train)

# Evaluating the model
y_pred = pipeline.predict(X_test)
accuracy_before_noise = accuracy_score(y_test, y_pred)

# Introducing more significant Gaussian noise to the test set (numerical features only)
X_test_noisy = X_test.copy()
numerical_indices = X_test_noisy.select_dtypes(include=['float64', 'int64']).columns
X_test_noisy[numerical_indices] += np.random.normal(0, 15, X_test_noisy[numerical_indices].shape)  

# Evaluating the model on noisy data
y_pred_noisy = pipeline.predict(X_test_noisy)
accuracy_after_noise = accuracy_score(y_test, y_pred_noisy)

print(f"Accuracy before noise: {accuracy_before_noise}")
print(f"Accuracy after noise: {accuracy_after_noise}")