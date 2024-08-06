import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "alcohol", "Maths.csv")
raw_data = pd.read_csv(raw_data_file)

# Known dataset with high-correlation between G1, G2 and G3 features
X = raw_data.drop(columns=['G3'])
y = raw_data['G3']

# Encoding categorical variables
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training a simple classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)

# Evaluating feature importance
perm_importance = permutation_importance(clf, X_train_scaled, y_train, n_repeats=10, random_state=42)
feature_importances = perm_importance.importances_mean
print(f"Feature importances: {feature_importances}")

# Data augmentation: Adding noise to reduce reliance on specific features
X_train_augmented = X_train.copy()
X_train_augmented['G1'] += np.random.normal(0, 0.1, X_train.shape[0])

# Training the model again with augmented data
clf.fit(X_train_augmented, y_train)

# Prediction and evaluation
y_pred = clf.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Domain adaptation techniques (example: reweighting samples)
weights = y_train.value_counts(normalize=True)[y_train]
clf.fit(X_train_scaled, y_train, sample_weight=weights)

# Predicting and evaluating with reweighted samples
y_pred = clf.predict(X_test_scaled)
print(f"Accuracy with reweighted samples: {accuracy_score(y_test, y_pred)}")

# Checking if the model still relies heavily on features or has learned to generalize better
perm_importance_post = permutation_importance(clf, X_train_scaled, y_train, n_repeats=10, random_state=42)
feature_importances_post = perm_importance_post.importances_mean
print(f"Feature importances after augmentation and reweighting: {feature_importances_post}")