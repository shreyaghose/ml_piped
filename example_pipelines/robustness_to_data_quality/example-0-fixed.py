import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "adult_data", "adult_data.csv")
data = pd.read_csv(raw_data_file)

# Define categorical and numerical features
X = data.drop('salary', axis=1)
y = data['salary']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['number']).columns.tolist()

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # More robust imputation
            ('scaler', RobustScaler())  # Scaling that is robust to outliers
        ]), numerical_features)
    ],
    remainder='drop'  # Drop any other columns not specified
)

# Create a pipeline that includes preprocessing and model fitting
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))  # Using Random Forest for sensitivity
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optionally: Train with noise augmentation to improve robustness
def add_noise(X, noise_level=15):
    noisy_data = X.copy()
    numerical_indices = noisy_data.select_dtypes(include=['float64', 'int64']).columns
    noisy_data[numerical_indices] += np.random.normal(0, noise_level, noisy_data[numerical_indices].shape)
    return noisy_data

# Augment the training data with noise
X_train_noisy = add_noise(X_train, noise_level=0.1)

# Train the model on augmented training data
pipeline.fit(X_train_noisy, y_train)

# Evaluate the model on the clean test set
y_pred = pipeline.predict(X_test)
accuracy_before_noise = accuracy_score(y_test, y_pred)

# Introduce more significant Gaussian noise to the test set (numerical features only)
X_test_noisy = X_test.copy()
numerical_indices = X_test_noisy.select_dtypes(include=['float64', 'int64']).columns
X_test_noisy[numerical_indices] += np.random.normal(0, 15, X_test_noisy[numerical_indices].shape)  

# Evaluate the model on noisy data
y_pred_noisy = pipeline.predict(X_test_noisy)
accuracy_after_noise = accuracy_score(y_test, y_pred_noisy)

print(f"Accuracy before noise: {accuracy_before_noise}")
print(f"Accuracy after noise: {accuracy_after_noise}")