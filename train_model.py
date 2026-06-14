"""Train and compare ML models for NayePankh donor conversion prediction."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join("data", "donor_data.csv")
OUTPUT_DIR = "outputs"


def main():
    df = pd.read_csv(DATA_PATH)

    print("=== Data Overview ===")
    print(df.describe())
    print(f"\nDonation rate: {df['donated'].mean() * 100:.1f}%")

    X = df.drop("donated", axis=1)
    y = df["donated"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_eda_plots(df)

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = []
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results.append(
            {"Model": name, "Accuracy": round(acc, 3), "F1": round(f1, 3), "AUC": round(auc, 3)}
        )

        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred))

    results_df = pd.DataFrame(results)
    print("\n=== Model Comparison ===")
    print(results_df.to_string(index=False))

    save_model_comparison(results_df)
    save_confusion_matrix(X_train, y_train, X_test, y_test)
    save_feature_importance(X_train, y_train, X.columns)

    print(f"\nSab graphs '{OUTPUT_DIR}/' folder me save ho gayi!")


def save_eda_plots(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.countplot(data=df, x="donated", ax=axes[0])
    axes[0].set_title("Donated vs Not Donated")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["No", "Yes"])

    sns.boxplot(data=df, x="donated", y="time_on_site_sec", ax=axes[1])
    axes[1].set_title("Time on Site by Donation")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_plots.png"), dpi=150)
    plt.close()
    print("Saved: outputs/eda_plots.png")


def save_model_comparison(results_df):
    results_df.set_index("Model")[["Accuracy", "F1", "AUC"]].plot(kind="bar", figsize=(8, 5))
    plt.title("Model Comparison - NayePankh Donor Prediction")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    print("Saved: outputs/model_comparison.png")


def save_confusion_matrix(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Saved: outputs/confusion_matrix.png")


def save_feature_importance(X_train, y_train, feature_names):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()

    print("\n=== Feature Importance ===")
    print(importances.sort_values(ascending=False).to_string())

    plt.figure(figsize=(8, 4))
    importances.plot(kind="barh")
    plt.title("Feature Importance - Donation ke factors")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("Saved: outputs/feature_importance.png")


if __name__ == "__main__":
    main()
