import os
import sys
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

# Using known biased dataset
X = data.drop('salary', axis=1)
y = data['salary']

# Convert target variable to numeric
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert categorical 'race' feature to numeric
race_encoder = LabelEncoder()
X['race'] = race_encoder.fit_transform(X['race'])

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Handle categorical data using one-hot encoding if necessary
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
X_train_encoded = encoder.fit_transform(X_train.select_dtypes(include=['object']))
X_test_encoded = encoder.transform(X_test.select_dtypes(include=['object']))

X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(X_train.select_dtypes(include=['object']).columns))
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(X_test.select_dtypes(include=['object']).columns))

# Combine encoded and non-encoded features
X_train_final = pd.concat([X_train.select_dtypes(exclude=['object']).reset_index(drop=True), X_train_encoded_df.reset_index(drop=True)], axis=1)
X_test_final = pd.concat([X_test.select_dtypes(exclude=['object']).reset_index(drop=True), X_test_encoded_df.reset_index(drop=True)], axis=1)

# Ensure all column names are strings
X_train_final.columns = X_train_final.columns.astype(str)
X_test_final.columns = X_test_final.columns.astype(str)

# Define the bias mitigator
model = LogisticRegression(max_iter=10000)
mitigator = ExponentiatedGradient(
    estimator=model,
    constraints=DemographicParity()
)

# 'race' is now numeric after encoding
X_train_final_race = X_train[['race']]
mitigator.fit(X_train_final, y_train, sensitive_features=X_train_final_race)

# Evaluate the model
y_pred = mitigator.predict(X_test_final)
print(classification_report(y_test, y_pred))