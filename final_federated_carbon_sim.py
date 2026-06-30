import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def run_federated_simulation():

    # ---------------- CREATE RESULTS FOLDER ----------------
    os.makedirs("results", exist_ok=True)

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

    # encode categorical columns
    categorical_cols = [1, 2, 3]

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        X_test[col] = le.transform(X_test[col])

    # scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    # ---------------- FEDERATED CLIENTS ----------------
    num_clients = 5

    client_data = np.array_split(X, num_clients)
    client_labels = np.array_split(y, num_clients)

    global_model = RandomForestClassifier(
        n_estimators=60,
        max_depth=10,
        random_state=42
    )

    round_accuracies = []
    executed_rounds = []
    carbon_levels = []
    decisions = []

    # ---------------- FEDERATED TRAINING ----------------
    for round_no in range(5):

        carbon_intensity = random.uniform(0, 1)
        carbon_levels.append(carbon_intensity)

        print(f"\nRound {round_no + 1} carbon level: {carbon_intensity:.4f}")

        if carbon_intensity > 0.6:

            print("⚠️ High carbon → skipping training round")

            decisions.append("Skip")

            continue

        decisions.append("Train")

        # local training
        local_models = []

        for i in range(num_clients):

            model = RandomForestClassifier(
                n_estimators=60,
                max_depth=10,
                random_state=42
            )

            model.fit(
                client_data[i],
                client_labels[i]
            )

            local_models.append(model)

        # simple aggregation
        sample_idx = np.random.choice(
            len(X),
            size=5000
        )

        global_model.fit(
            X[sample_idx],
            y.iloc[sample_idx]
        )

        y_pred = global_model.predict(X_test)

        acc = accuracy_score(
            y_test,
            y_pred
        )

        print(f"Global accuracy: {acc:.4f}")

        round_accuracies.append(acc * 100)
        executed_rounds.append(round_no + 1)

    # ---------------- SAVE RESULTS ----------------
    results = []

    accuracy_idx = 0

    for i in range(5):

        if decisions[i] == "Train":
            acc = round_accuracies[accuracy_idx]
            accuracy_idx += 1
        else:
            acc = None

        results.append({
            "Round": i + 1,
            "Carbon_Intensity": round(
                carbon_levels[i], 4
            ),
            "Decision": decisions[i],
            "Accuracy": acc
        })

    df = pd.DataFrame(results)

    df.to_csv(
        "results/carbon_results.csv",
        index=False
    )

    # ---------------- FIGURE 9 ----------------
    colors = []

    for d in decisions:
        if d == "Train":
            colors.append("green")
        else:
            colors.append("gray")

    plt.figure(figsize=(8, 5))

    plt.bar(
        range(1, 6),
        carbon_levels,
        color=colors
    )

    plt.axhline(
        y=0.6,
        color="red",
        linestyle="--",
        label="Threshold = 0.60"
    )

    plt.xticks(range(1, 6))
    plt.xlabel("Training Round")
    plt.ylabel("Carbon Intensity")
    plt.title("Carbon-Aware Scheduling Decisions")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/carbon_intensity_chart.png",
        dpi=300
    )

    plt.close()

    # ---------------- FIGURE 10 ----------------
    if len(round_accuracies) > 0:

        plt.figure(figsize=(8, 5))

        bars = plt.bar(
            [f"Round {r}" for r in executed_rounds],
            round_accuracies
        )

        plt.title(
            "Global Accuracy of Executed Training Rounds"
        )

        plt.xlabel("Training Round")
        plt.ylabel("Accuracy (%)")

        plt.ylim(75, 82)

        plt.grid(
            axis="y",
            linestyle="--",
            alpha=0.7
        )

        for bar in bars:
            height = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.05,
                f"{height:.2f}%",
                ha="center"
            )

        plt.tight_layout()

        plt.savefig(
            "results/accuracy_bar_chart.png",
            dpi=300
        )

        plt.close()

    print("\nResults saved in results/")
    print(df)

    return df

if __name__ == "__main__":
    run_federated_simulation()