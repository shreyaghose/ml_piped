import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

# Identify potential proxy attributes (modify this method as needed)
def identify_proxy_attributes(df, protected_attributes):
    proxy_attributes = set()
    
    # Placeholder for proxy attributes identification logic
    # Since correlation doesn't work with categorical data, use domain knowledge or other methods.
    
    # Example: For simplicity, we will skip proxy identification in this example
    return list(proxy_attributes)

# List of protected attributes
protected_attributes = ['race', 'gender']

# Split the dataset into train and test sets
# Splitting before processing to ensure transformations are applied separately
X_train, X_test, y_train, y_test = train_test_split(data.drop('salary', axis=1), data['salary'], test_size=0.2, random_state=42)

# Identify proxy attributes after the split (currently an empty list)
proxy_attributes = identify_proxy_attributes(X_train, protected_attributes)

# Remove proxy attributes from the feature set
features_to_include = [col for col in X_train.columns if col not in protected_attributes + proxy_attributes]
X_train = X_train[features_to_include]
X_test = X_test[features_to_include]

# Identify categorical columns
categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()

# Create a column transformer with one-hot encoding for categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ],
    remainder='passthrough'
)

# Create a pipeline with preprocessing and the classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Output the removed proxy attributes for transparency
print(f"Removed proxy attributes: {proxy_attributes}")