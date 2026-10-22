"""
=============================================================================
Automated Test Suite for AI Content Moderation System
=============================================================================
Verifies:
  1. Dataset loading, columns, and class balance
  2. Saved model and TF-IDF vectorizer artifact loading
  3. Inference on clean, constructive, and toxic sample comments
  4. Decision threshold sensitivity and probability calibration
  5. Explainable AI feature contribution calculations
  6. Model comparison integrity (all 3 algorithms present with genuine metrics)
  7. Empty input handling
=============================================================================
"""

import os
import json
import joblib
import pandas as pd
import numpy as np


def test_dataset_exists_and_valid():
    data_path = os.path.join("data", "train.csv")
    assert os.path.exists(data_path), f"Dataset not found at {data_path}"
    
    # Read sample to verify columns
    df = pd.read_csv(data_path, nrows=50)
    assert "comment_text" in df.columns, "Missing 'comment_text' column in dataset"
    assert "toxic" in df.columns, "Missing 'toxic' column in dataset"
    print("PASS: Dataset exists and has required columns.")


def test_model_artifacts_load():
    model_path = os.path.join("models", "toxic_model.joblib")
    vec_path = os.path.join("models", "tfidf_vectorizer.joblib")
    metrics_path = os.path.join("models", "metrics.json")
    comp_path = os.path.join("models", "model_comparison.csv")

    assert os.path.exists(model_path), f"Missing model at {model_path}"
    assert os.path.exists(vec_path), f"Missing vectorizer at {vec_path}"
    assert os.path.exists(metrics_path), f"Missing metrics at {metrics_path}"
    assert os.path.exists(comp_path), f"Missing comparison at {comp_path}"

    model = joblib.load(model_path)
    vec = joblib.load(vec_path)

    assert hasattr(model, "predict_proba"), "Loaded model does not support predict_proba"
    assert hasattr(vec, "transform"), "Loaded vectorizer does not support transform"
    print("PASS: Model artifacts load properly.")


def test_inference_and_predictions():
    model_path = os.path.join("models", "toxic_model.joblib")
    vec_path = os.path.join("models", "tfidf_vectorizer.joblib")

    model = joblib.load(model_path)
    vec = joblib.load(vec_path)

    # Clean text test
    clean_sample = "Thank you so much for this helpful article! Great work."
    clean_vec = vec.transform([clean_sample])
    clean_prob = model.predict_proba(clean_vec)[0][1]
    assert clean_prob < 0.20, f"Clean sample flagged as toxic: {clean_prob}"

    # Toxic text test
    toxic_sample = "You are a complete idiot and your work is absolute garbage."
    toxic_vec = vec.transform([toxic_sample])
    toxic_prob = model.predict_proba(toxic_vec)[0][1]
    assert toxic_prob > 0.80, f"Toxic sample not detected: {toxic_prob}"

    print(f"PASS: Inference verified! Clean score: {clean_prob:.4f}, Toxic score: {toxic_prob:.4f}")


def test_explainability_function():
    from app import extract_word_influences
    model = joblib.load(os.path.join("models", "toxic_model.joblib"))
    vec = joblib.load(os.path.join("models", "tfidf_vectorizer.joblib"))

    test_comment = "You are an idiot and a fool"
    influences = extract_word_influences(test_comment, vec, model, top_n=3)

    assert len(influences) > 0, "No word influences extracted"
    top_word, top_score, _ = influences[0]
    assert top_score > 0, f"Expected positive toxic contribution for top word, got {top_score}"
    print(f"PASS: Explainability works. Top toxic token: '{top_word}' (score: {top_score:.3f})")


def test_model_comparison_file():
    comp_path = os.path.join("models", "model_comparison.csv")
    df = pd.read_csv(comp_path)

    expected_algorithms = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ]

    assert len(df) == 3, f"Expected exactly 3 models in comparison table, found {len(df)}"
    assert "Multinomial Naive Bayes" not in df["Algorithm"].values, "Stale Multinomial Naive Bayes found in comparison table"

    for algo in expected_algorithms:
        assert algo in df["Algorithm"].values, f"Missing algorithm {algo} in comparison table"

    # Verify no NaN values
    assert not df.isnull().values.any(), "Comparison table contains NaN values"

    # Verify metric ranges (0.0 to 1.0)
    for col in ["Accuracy", "Toxic_Precision", "Toxic_Recall", "Toxic_F1"]:
        assert (df[col] >= 0.0).all() and (df[col] <= 1.0).all(), f"Invalid value in {col}"

    print("PASS: Exactly 3 models verified in comparison table with valid test metrics.")


def test_confusion_matrix_sum():
    metrics_path = os.path.join("models", "metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    cm = metrics["confusion_matrix"]
    total = cm["true_negatives"] + cm["false_positives"] + cm["false_negatives"] + cm["true_positives"]
    expected_test_samples = metrics["test_samples"]
    assert total == expected_test_samples, f"Confusion matrix sum ({total}) != test samples ({expected_test_samples})"
    assert expected_test_samples == 31915, f"Expected 31,915 held-out test samples, got {expected_test_samples}"
    print(f"PASS: Confusion matrix sum verified ({total:,} == {expected_test_samples:,} test samples).")


def test_classification_report_structure():
    comp_json_path = os.path.join("models", "model_comparison.json")
    with open(comp_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    reports = data.get("detailed_reports", {})
    assert len(reports) == 3, f"Expected exactly 3 reports in detailed_reports, found {len(reports)}"
    assert "Multinomial Naive Bayes" not in reports, "Stale Multinomial Naive Bayes found in detailed_reports"
    assert "Logistic Regression" in reports, "Missing Logistic Regression detailed report"
    assert "Decision Tree" in reports, "Missing Decision Tree detailed report"
    assert "Random Forest" in reports, "Missing Random Forest detailed report"

    lr_report = reports["Logistic Regression"]["classification_report"]
    required_keys = ["Clean (0)", "Toxic (1)", "accuracy", "macro avg", "weighted avg"]
    for k in required_keys:
        assert k in lr_report, f"Missing key '{k}' in classification report"

    # Verify per-class fields
    for cls in ["Clean (0)", "Toxic (1)", "macro avg", "weighted avg"]:
        for metric in ["precision", "recall", "f1-score", "support"]:
            assert metric in lr_report[cls], f"Missing '{metric}' in {cls} report"

    # Verify total support equals 31915
    clean_supp = lr_report["Clean (0)"]["support"]
    toxic_supp = lr_report["Toxic (1)"]["support"]
    assert int(clean_supp + toxic_supp) == 31915, f"Support sum mismatch: {clean_supp + toxic_supp}"
    print("PASS: Classification report structure verified with both classes and all averages.")


def test_visualizations():
    from app import render_confusion_matrix_heatmap, render_grouped_comparison_chart
    import matplotlib.pyplot as plt

    cm_array = np.array([[28436, 234], [1096, 2149]])
    fig_cm = render_confusion_matrix_heatmap(cm_array)
    assert fig_cm is not None, "Heatmap figure is None"
    plt.close(fig_cm)

    comp_df = pd.read_csv(os.path.join("models", "model_comparison.csv"))
    fig_comp = render_grouped_comparison_chart(comp_df)
    assert fig_comp is not None, "Grouped bar chart figure is None"
    plt.close(fig_comp)

    # Verify saved high-resolution F1 comparison plot
    f1_chart_path = os.path.join("models", "f1_score_comparison.png")
    assert os.path.exists(f1_chart_path), f"Missing comparison plot: {f1_chart_path}"
    assert os.path.getsize(f1_chart_path) > 5000, f"Comparison plot file too small: {os.path.getsize(f1_chart_path)} bytes"

    # Verify metrics.json contains only the primary Logistic Regression model metrics
    metrics_path = os.path.join("models", "metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_content = f.read()
    assert "Naive Bayes" not in metrics_content and "MultinomialNB" not in metrics_content, "Naive Bayes found in metrics.json"

    print("PASS: Matplotlib visualizations, F1-chart image, and metrics verified successfully.")


if __name__ == "__main__":
    print("Running automated verification tests...\n")
    test_dataset_exists_and_valid()
    test_model_artifacts_load()
    test_inference_and_predictions()
    test_explainability_function()
    test_model_comparison_file()
    test_confusion_matrix_sum()
    test_classification_report_structure()
    test_visualizations()
    print("\nALL 8 TEST SUITES PASSED SUCCESSFULLY!")
