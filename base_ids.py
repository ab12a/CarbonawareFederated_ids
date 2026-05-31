import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
train = pd.read_csv("data/KDDTrain+.txt", header=None)
test = pd.read_csv("data/KDDTest+.txt", header=None)

# Features and labels
X_train = train.iloc[:, :-2]
y_train = train.iloc[:, -2]

X_test = test.iloc[:, :-2]
y_test = test.iloc[:, -2]

# Convert attack types to binary
y_train = y_train.apply(lambda x: 0 if x == "normal" else 1)
y_test = y_test.apply(lambda x: 0 if x == "normal" else 1)

# Encode categorical features
categorical_cols = [1, 2, 3]   # protocol, service, flag

for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=80, max_depth=12, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

with open("results/baseline_results.txt", "w") as f:
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")