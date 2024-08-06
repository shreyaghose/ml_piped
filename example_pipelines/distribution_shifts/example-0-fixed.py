import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from sklearn.kernel_ridge import KernelRidge

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import get_project_root

# Getting the project root
project_root = get_project_root()

# Getting the raw data file
raw_data_file = os.path.join(project_root, "datasets", "titanic", "data.csv")
data = pd.read_csv(raw_data_file)

# Dropping irrelevant columns
data = data.drop(['Name', 'Cabin', 'Ticket'], axis=1)

# Artificially creating a covariate shift by sampling classes unequally
df_class_0 = data[data['Survived'] == 0].sample(frac=0.6, random_state=42)
df_class_1 = data[data['Survived'] == 1]

df_shifted = pd.concat([df_class_0, df_class_1])

# Handling missing values
df_shifted['Age'] = df_shifted['Age'].fillna(df_shifted['Age'].median())
df_shifted['Embarked'] = df_shifted['Embarked'].fillna(df_shifted['Embarked'].mode()[0])
df_shifted['Fare'] = df_shifted['Age'].fillna(df_shifted['Fare'].median())

# Encoding categorical variables
le = LabelEncoder()
df_shifted['Sex'] = le.fit_transform(df_shifted['Sex'])
df_shifted['Embarked'] = le.fit_transform(df_shifted['Embarked'])

# Splitting the dataset into train and test sets
X = df_shifted.drop(['Survived'], axis=1)
y = df_shifted['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying scaling to handle potential differences in feature distributions
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Checking for distribution shift using a simple model
clf_shift_detection = KernelRidge(kernel='rbf')
clf_shift_detection.fit(X_train_scaled, y_train)
train_preds = clf_shift_detection.predict(X_train_scaled)
test_preds = clf_shift_detection.predict(X_test_scaled)

# If there is a significant difference in the predictions, a shift is detected
train_pred_mean = train_preds.mean()
test_pred_mean = test_preds.mean()

if abs(train_pred_mean - test_pred_mean) > 0.1:  # threshold for detecting shift
    print("Covariate shift detected, applying reweighting techniques")
    
    # Applying reweighting technique to mitigate the shift. Reweighting using importance sampling
    X_combined = pd.concat([X_train, X_test])
    y_combined = pd.concat([y_train, y_test])
    
    weights = y_combined.value_counts(normalize=True)[y_combined] / y_combined.value_counts(normalize=True)[y_train]
    X_train_resampled, y_train_resampled = resample(X_train, y_train, replace=True, n_samples=len(y_train), random_state=42, stratify=y_train, weights=weights)
else:
    X_train_resampled, y_train_resampled = X_train, y_train

# Training a simple classifier with resampled data
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_resampled, y_train_resampled)

# Prediction and evaluation
y_pred = clf.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")