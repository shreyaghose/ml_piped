import sys
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "diabetes_indicator", "5050_split.csv")
data = pd.read_csv(raw_data_file)

# Down-sample the data before cross-validation
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(data.drop('Diabetes_binary', axis=1), data['Diabetes_binary'])

# Feature selection before cross-validation
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X_resampled, y_resampled)

# Cross-validation
clf = RandomForestClassifier(random_state=42)
scores = cross_val_score(clf, X_selected, y_resampled, cv=5)

print("Cross-validation scores:", scores)
