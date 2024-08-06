import os
import sys
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "diabetes_indicator", "5050_split.csv")
data = pd.read_csv(raw_data_file)

# Defining pipeline with under-sampling and feature selection
pipeline = ImbPipeline([
    ('sampling', RandomUnderSampler(random_state=42)),
    ('feature_selection', SelectKBest(f_classif, k=10)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Stratified cross-validation
cv = StratifiedKFold(n_splits=5)

# Nested cross-validation for hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid_search.fit(data.drop('Diabetes_binary', axis=1), data['Diabetes_binary'])

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)