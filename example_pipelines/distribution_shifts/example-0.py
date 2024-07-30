import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "titanic", "data.csv")
data = pd.read_csv(raw_data_file)
data = data.drop(['Name', 'Cabin', 'Ticket'], axis=1)

# Artificially create a covariate shift by sampling classes unequally
df_class_0 = data[data['Survived'] == 0].sample(frac=0.6, random_state=42)
df_class_1 = data[data['Survived'] == 1]

df_shifted = pd.concat([df_class_0, df_class_1])

# Handle missing values
df_shifted['Age'] = df_shifted['Age'].fillna(df_shifted['Age'].median())
df_shifted['Embarked'] = df_shifted['Embarked'].fillna(df_shifted['Embarked'].mode()[0])
df_shifted['Fare'] = df_shifted['Age'].fillna(df_shifted['Fare'].median())

# Encode categorical variables
le = LabelEncoder()
df_shifted['Sex'] = le.fit_transform(df_shifted['Sex'])
df_shifted['Embarked'] = le.fit_transform(df_shifted['Embarked'])

# Check for any remaining missing values after encoding
print("Missing values after encoding:")
print(df_shifted.isnull().sum())

# Split the dataset into train and test sets
X = df_shifted.drop('Survived', axis=1)
y = df_shifted['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check for any remaining missing values in train and test sets
print("Missing values in X_train:")
print(X_train.isnull().sum())
print("Missing values in X_test:")
print(X_test.isnull().sum())

# Train a simple classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report: {classification_report(y_test, y_pred)}")