"""
=============================================================================
AI Content Moderation System - Comprehensive Model Comparison & Training
=============================================================================
This script trains and compares THREE machine learning classification algorithms:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. Random Forest Classifier

[STRICT RULE ENFORCED]: No SVM / Support Vector Machine models are used.

Key Pipeline Steps:
  - Loads real Jigsaw Toxic Comment Classification Challenge data (159,571 rows).
  - Target: Binary classification (0 = Clean/Safe, 1 = Toxic/Harmful).
  - Stratified 80:20 train/test split with fixed random seed (seed=42).
  - Fits TF-IDF vectorizer (unigrams + bigrams, 25k features) ONLY on training data.
  - Trains all 3 models and measures exact training time.
  - Evaluates every model on the 31,915 held-out test comments.
  - Computes genuine Accuracy, Precision, Recall, F1-Score (toxic class), and Confusion Matrix.
  - Compares results, saves comparison table (CSV & JSON), and generates F1 comparison plot.
  - Automatically identifies and saves the preferred model, vectorizer, and metrics.
=============================================================================
"""

import os
import re
import json
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script rendering
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Paths and directories
DATA_PATH = os.path.join("data", "train.csv")
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "toxic_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
COMPARISON_CSV_PATH = os.path.join(MODELS_DIR, "model_comparison.csv")
COMPARISON_JSON_PATH = os.path.join(MODELS_DIR, "model_comparison.json")
COMPARISON_CHART_PATH = os.path.join(MODELS_DIR, "f1_score_comparison.png")


def clean_text(text: str) -> str:
    """
    Cleans raw text comments:
      - Lowercase conversion
      - Removes web URLs
      - Removes non-alphabetic characters (numbers, punctuation)
      - Normalizes spaces
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset():
    """
    Loads the real Jigsaw Toxic Comment dataset.
    If the file is not present locally, triggers download_data.py.
    """
    if not os.path.exists(DATA_PATH):
        print(f"[INFO] Dataset not found at '{DATA_PATH}'. Initiating download...")
        import download_data
        download_data.download_dataset()

    print(f"\n[STEP 1/5] Loading real dataset from '{DATA_PATH}'...")
    df = pd.read_csv(DATA_PATH)
    print(f"-> Total records loaded: {len(df):,}")

    df["comment_text"] = df["comment_text"].fillna("")

    # Multi-label toxicity columns from Jigsaw challenge:
    # A comment is considered toxic (1) if ANY toxicity category is flagged
    category_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    existing_cols = [col for col in category_columns if col in df.columns]

    if existing_cols:
        df["label"] = (df[existing_cols].sum(axis=1) > 0).astype(int)
    else:
        target_col = "toxic" if "toxic" in df.columns else "label"
        df["label"] = df[target_col].astype(int)

    toxic_count = int(df["label"].sum())
    clean_count = int(len(df) - toxic_count)
    print(f"-> Class Distribution: {clean_count:,} Clean ({clean_count/len(df)*100:.1f}%), "
          f"{toxic_count:,} Toxic ({toxic_count/len(df)*100:.1f}%)")

    return df


def plot_f1_comparison(comparison_df, output_path):
    """
    Generates and saves a clean bar chart comparing toxic-class F1-scores.
    """
    try:
        plt.figure(figsize=(9, 5))
        colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]
        bars = plt.bar(
            comparison_df["Algorithm"],
            comparison_df["Toxic_F1"] * 100,
            color=colors[:len(comparison_df)],
            edgecolor="black",
            width=0.55
        )
        plt.title("Model Comparison: Toxic-Class F1-Score (Test Set)", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Machine Learning Algorithm", fontsize=11, labelpad=8)
        plt.ylabel("F1-Score (%)", fontsize=11, labelpad=8)
        plt.ylim(0, 100)
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1.5,
                f"{height:.2f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"-> F1-Score comparison chart saved: {output_path}")
    except Exception as e:
        print(f"[WARNING] Could not save comparison plot: {e}")


def train_and_compare_models():
    """
    Trains and compares all 3 algorithms on identical stratified test/train splits.
    """
    total_start_time = time.time()
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load Data
    df = load_dataset()

    # 2. Stratified Train/Test Split (80% Train, 20% Test)
    print("\n[STEP 2/5] Creating Stratified 80:20 Train/Test Split (random_state=42)...")
    X = df["comment_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"-> Training Set: {len(X_train):,} samples (Clean: {(y_train==0).sum():,}, Toxic: {(y_train==1).sum():,})")
    print(f"-> Held-out Test Set: {len(X_test):,} samples (Clean: {(y_test==0).sum():,}, Toxic: {(y_test==1).sum():,})")

    # 3. TF-IDF Feature Extraction (Fit ONLY on training data)
    print("\n[STEP 3/5] Extracting TF-IDF Features (Fit strictly on train set)...")
    tfidf = TfidfVectorizer(
        max_features=25000,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    t0 = time.time()
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    print(f"-> TF-IDF extraction complete in {time.time() - t0:.2f}s (Vocabulary: {len(tfidf.vocabulary_):,} features)")

    # 4. Define the 3 Algorithms to compare (Strictly NO SVMs)
    models_to_evaluate = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=3.0,
            solver="liblinear",
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=30,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=40,
            max_depth=None,
            min_samples_split=10,
            n_jobs=-1,
            random_state=42
        )
    }

    # 5. Train and Evaluate each model
    print("\n[STEP 4/5] Training and Evaluating All 3 Classifiers on Held-Out Test Set...")
    print("=" * 80)
    print(f"{'Algorithm':<26} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Train Time':<10}")
    print("=" * 80)

    results_list = []
    detailed_reports = {}
    fitted_models = {}

    for name, model in models_to_evaluate.items():
        # Train
        t_start = time.time()
        model.fit(X_train_vec, y_train)
        train_time = time.time() - t_start
        fitted_models[name] = model

        # Evaluate on unseen held-out test data
        y_pred = model.predict(X_test_vec)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]

        results_list.append({
            "Algorithm": name,
            "Accuracy": round(acc, 4),
            "Toxic_Precision": round(prec, 4),
            "Toxic_Recall": round(rec, 4),
            "Toxic_F1": round(f1, 4),
            "Training_Time_Sec": round(train_time, 2),
            "True_Negatives": tn,
            "False_Positives": fp,
            "False_Negatives": fn,
            "True_Positives": tp
        })

        detailed_reports[name] = {
            "classification_report": classification_report(
                y_test, y_pred, target_names=["Clean (0)", "Toxic (1)"], digits=4, output_dict=True, zero_division=0
            ),
            "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
        }

        print(f"{name:<26} | {acc*100:>8.2f}% | {prec*100:>8.2f}% | {rec*100:>8.2f}% | {f1*100:>8.2f}% | {train_time:>8.2f}s")

    print("=" * 80)

    # 6. Save Comparison Table
    comparison_df = pd.DataFrame(results_list)
    comparison_df.to_csv(COMPARISON_CSV_PATH, index=False)
    print(f"\n[STEP 5/5] Comparison table saved to '{COMPARISON_CSV_PATH}'")

    with open(COMPARISON_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "comparison": results_list,
            "detailed_reports": detailed_reports
        }, f, indent=4)
    print(f"-> Detailed comparison JSON saved to '{COMPARISON_JSON_PATH}'")

    # Generate F1 Bar Chart
    plot_f1_comparison(comparison_df, COMPARISON_CHART_PATH)

    # 7. Model Selection based on toxic-class F1-score (harmonic mean of precision and recall)
    # Sort descending by Toxic_F1
    best_row = comparison_df.sort_values(by="Toxic_F1", ascending=False).iloc[0]
    best_algorithm_name = best_row["Algorithm"]
    best_model = fitted_models[best_algorithm_name]

    print("\n" + "=" * 65)
    print(f"[PREFERRED MODEL SELECTED]: {best_algorithm_name.upper()}")
    print("=" * 65)
    print(f"Selection Criterion: Highest Toxic-Class F1-Score ({best_row['Toxic_F1']*100:.2f}%)")
    print(f"-> Accuracy:        {best_row['Accuracy']*100:.2f}%")
    print(f"-> Toxic Precision: {best_row['Toxic_Precision']*100:.2f}% (Low false positives)")
    print(f"-> Toxic Recall:    {best_row['Toxic_Recall']*100:.2f}% (Catches toxic comments)")
    print(f"-> Training Time:   {best_row['Training_Time_Sec']:.2f} seconds")
    print(f"-> Confusion Matrix:")
    print(f"   TN: {best_row['True_Negatives']:,}  |  FP: {best_row['False_Positives']:,}")
    print(f"   FN: {best_row['False_Negatives']:,}  |  TP: {best_row['True_Positives']:,}")
    print("=" * 65)

    # 8. Save Selected Model, Vectorizer, and Metrics
    print(f"\n[SAVING] Serializing preferred model and vectorizer...")
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(tfidf, VECTORIZER_PATH)

    final_metrics = {
        "selected_model": best_algorithm_name,
        "dataset_name": "Jigsaw Toxic Comment Classification Challenge",
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": best_row["Accuracy"],
        "precision": best_row["Toxic_Precision"],
        "recall": best_row["Toxic_Recall"],
        "f1_score": best_row["Toxic_F1"],
        "confusion_matrix": {
            "true_negatives": int(best_row["True_Negatives"]),
            "false_positives": int(best_row["False_Positives"]),
            "false_negatives": int(best_row["False_Negatives"]),
            "true_positives": int(best_row["True_Positives"])
        },
        "training_time_seconds": best_row["Training_Time_Sec"],
        "total_pipeline_time": round(time.time() - total_start_time, 2)
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=4)

    print(f"-> Model saved:      {MODEL_PATH}")
    print(f"-> Vectorizer saved: {VECTORIZER_PATH}")
    print(f"-> Metrics saved:    {METRICS_PATH}")
    print(f"\nTraining pipeline completed successfully in {time.time() - total_start_time:.2f} seconds.")


if __name__ == "__main__":
    train_and_compare_models()
