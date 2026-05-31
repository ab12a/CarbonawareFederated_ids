import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------- LOAD DATA ----------------
train = pd.read_csv("data/KDDTrain+.txt", header=None)
test = pd.read_csv("data/KDDTest+.txt", header=None)

X = train.iloc[:, :-2]
y = train.iloc[:, -2]

X_test = test.iloc[:, :-2]
y_test = test.iloc[:, -2]

# binary labels
y = y.apply(lambda x: 0 if x == "normal" else 1)
y_test = y_test.apply(lambda x: 0 if x == "normal" else 1)

# encode categorical
categorical_cols = [1,2,3]
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])

# scale
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# ------------- SIMULATE IoT CLIENTS -------------
num_clients = 5
client_data = np.array_split(X, num_clients)
client_labels = np.array_split(y, num_clients)

global_model = RandomForestClassifier(n_estimators=60, max_depth=10)

round_accuracies = []

# ------------- FEDERATED TRAINING -------------
for round in range(5):

    # simulate carbon intensity (0 clean → 1 dirty)
    carbon_intensity = random.uniform(0,1)

    print(f"\nRound {round+1} carbon level:", carbon_intensity)

    if carbon_intensity > 0.6:
        print("⚠️ High carbon → skipping training round")
        continue

    local_models = []

    for i in range(num_clients):
        model = RandomForestClassifier(n_estimators=60, max_depth=10)
        model.fit(client_data[i], client_labels[i])
        local_models.append(model)

    # simple aggregation: retrain global on combined small sample
    sample_idx = np.random.choice(len(X), size=5000)
    global_model.fit(X[sample_idx], y.iloc[sample_idx])

    y_pred = global_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Global accuracy:", acc)
    round_accuracies.append(acc)

print("\nFinal accuracies per training round:", round_accuracies)

with open("results/federated_results.txt", "w") as f:
    for acc in round_accuracies:
        f.write(f"{acc}\n")