# fixed_pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the data
data = pd.read_csv('data.csv')

# Handle missing values and standardize the features
imputer = SimpleImputer(strategy='mean')  # Imputation for missing values
scaler = StandardScaler()  # Scaling for standardization

# Apply imputation and scaling
X = data.drop('target', axis=1)
y = data['target']
X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)

# Apply stratified sampling to ensure representational balance
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))