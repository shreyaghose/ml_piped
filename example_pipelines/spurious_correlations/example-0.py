import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "diabetes_indicator", "binary_health_indicators.csv")
data = pd.read_csv(raw_data_file)

# Split dataset into features and target
X = data.drop(columns=['Diabetes_binary'])
y = data['Diabetes_binary']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification report: {classification_report(y_test, y_pred)}")