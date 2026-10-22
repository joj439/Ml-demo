"""
=============================================================================
AI Content Moderation System - Streamlit Web Dashboard
=============================================================================
An interactive, production-ready machine learning dashboard for real-time
content moderation, model benchmarking, and performance evaluation.

Tech Stack:
  - Python, Streamlit, Pandas, Scikit-learn, TF-IDF, Logistic Regression, Joblib

Models Benchmarked:
  1. Logistic Regression (Selected Production Model - F1: 76.37%)
  2. Decision Tree Classifier
  3. Random Forest Classifier
=============================================================================
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Configure the Streamlit webpage
st.set_page_config(
    page_title="AI Content Moderation System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File Paths
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "toxic_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
COMPARISON_CSV_PATH = os.path.join(MODELS_DIR, "model_comparison.csv")
COMPARISON_JSON_PATH = os.path.join(MODELS_DIR, "model_comparison.json")
COMPARISON_CHART_PATH = os.path.join(MODELS_DIR, "f1_score_comparison.png")


CANONICAL_MODELS = ["Logistic Regression", "Decision Tree", "Random Forest"]


@st.cache_resource
def _load_production_artifacts_cached(model_mtime, vec_mtime):
    """
    Cached artifact loader that automatically invalidates whenever model
    files on disk are updated or retrained.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except Exception as e:
        st.sidebar.error(f"Error loading model artifacts: {e}")
        return None, None


def load_production_artifacts():
    """
    Wrapper that passes file modification timestamps to the cached loader
    to guarantee instant cache invalidation upon model retraining.
    """
    m_time = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0.0
    v_time = os.path.getmtime(VECTORIZER_PATH) if os.path.exists(VECTORIZER_PATH) else 0.0
    return _load_production_artifacts_cached(m_time, v_time)


@st.cache_data
def _load_metrics_and_comparison_cached(m_time, c_time, j_time):
    """
    Cached data loader keyed on file modification timestamps.
    Enforces strict canonical 3-model filtering on both CSV and JSON data.
    """
    metrics = None
    comparison_df = None
    comparison_json = None

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    if os.path.exists(COMPARISON_CSV_PATH):
        raw_df = pd.read_csv(COMPARISON_CSV_PATH)
        if "Algorithm" in raw_df.columns:
            # Strictly filter to the 3 approved benchmark algorithms
            comparison_df = raw_df[raw_df["Algorithm"].isin(CANONICAL_MODELS)].reset_index(drop=True)
        else:
            comparison_df = raw_df

    if os.path.exists(COMPARISON_JSON_PATH):
        with open(COMPARISON_JSON_PATH, "r", encoding="utf-8") as f:
            raw_json = json.load(f)
        comparison_json = {}
        if "comparison" in raw_json:
            comparison_json["comparison"] = [
                item for item in raw_json["comparison"]
                if item.get("Algorithm") in CANONICAL_MODELS
            ]
        if "detailed_reports" in raw_json:
            comparison_json["detailed_reports"] = {
                k: v for k, v in raw_json["detailed_reports"].items()
                if k in CANONICAL_MODELS
            }

    return metrics, comparison_df, comparison_json


def load_metrics_and_comparison():
    """
    Wrapper that retrieves file modification timestamps to ensure the
    Streamlit cache is invalidated immediately when evaluation files change.
    """
    m_time = os.path.getmtime(METRICS_PATH) if os.path.exists(METRICS_PATH) else 0.0
    c_time = os.path.getmtime(COMPARISON_CSV_PATH) if os.path.exists(COMPARISON_CSV_PATH) else 0.0
    j_time = os.path.getmtime(COMPARISON_JSON_PATH) if os.path.exists(COMPARISON_JSON_PATH) else 0.0
    return _load_metrics_and_comparison_cached(m_time, c_time, j_time)


def extract_word_influences(text, vectorizer, model, top_n=6):
    """
    Calculates feature contribution for words present in the comment.
    For Logistic Regression: score = coefficient * tfidf_value (linear log-odds contribution).
    Higher positive scores indicate words strongly associated with toxicity.
    """
    if not hasattr(model, "coef_"):
        return []

    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]

    vec = vectorizer.transform([text])
    feature_indices = vec.nonzero()[1]

    word_scores = []
    for idx in feature_indices:
        word = feature_names[idx]
        tfidf_val = vec[0, idx]
        contribution = coef[idx] * tfidf_val
        word_scores.append((word, contribution, tfidf_val))

    # Sort descending by contribution (most toxic words first)
    word_scores.sort(key=lambda x: x[1], reverse=True)
    return word_scores[:top_n]


def render_confusion_matrix_heatmap(cm_array, class_names=["Non-toxic", "Toxic"]):
    """
    Renders a properly labelled confusion matrix heatmap using Matplotlib.
    - X-axis: Predicted label
    - Y-axis: Actual label
    - Class labels: Non-toxic (0) and Toxic (1)
    - Displays integer counts inside every cell with contrasting text and color bar.
    - Class order: [0, 1] (0 = Non-toxic, 1 = Toxic)
    """
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(cm_array, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=10, fontweight="bold")
    ax.set_yticklabels(class_names, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted label", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Actual label", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Confusion Matrix Heatmap", fontsize=12, fontweight="bold", pad=12)

    thresh = cm_array.max() / 2.0
    for i in range(2):
        for j in range(2):
            count = int(cm_array[i, j])
            text_color = "white" if count > thresh else "black"
            subtext = ""
            if i == 0 and j == 0:
                subtext = "\n(TN)"
            elif i == 0 and j == 1:
                subtext = "\n(FP)"
            elif i == 1 and j == 0:
                subtext = "\n(FN)"
            elif i == 1 and j == 1:
                subtext = "\n(TP)"
            ax.text(
                j, i, f"{count:,}{subtext}",
                ha="center", va="center",
                color=text_color, fontsize=11, fontweight="bold"
            )

    plt.tight_layout()
    return fig


def render_grouped_comparison_chart(comparison_df):
    """
    Renders a grouped bar chart comparing Accuracy, Toxic-Class Precision,
    Toxic-Class Recall, and Toxic-Class F1-Score across all three algorithms.
    """
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    algorithms = comparison_df["Algorithm"].tolist()
    x = np.arange(len(algorithms))
    bar_width = 0.18

    acc = comparison_df["Accuracy"] * 100
    prec = comparison_df["Toxic_Precision"] * 100
    rec = comparison_df["Toxic_Recall"] * 100
    f1 = comparison_df["Toxic_F1"] * 100

    r1 = ax.bar(x - 1.5 * bar_width, acc, bar_width, label="Accuracy", color="#1f77b4", edgecolor="black", linewidth=0.5)
    r2 = ax.bar(x - 0.5 * bar_width, prec, bar_width, label="Toxic Precision", color="#2ca02c", edgecolor="black", linewidth=0.5)
    r3 = ax.bar(x + 0.5 * bar_width, rec, bar_width, label="Toxic Recall", color="#ff7f0e", edgecolor="black", linewidth=0.5)
    r4 = ax.bar(x + 1.5 * bar_width, f1, bar_width, label="Toxic F1-Score", color="#d62728", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Comprehensive 3-Model Metric Comparison (Held-Out Test Set)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 112)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    for bars in [r1, r2, r3, r4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7.5, fontweight="bold"
            )

    plt.tight_layout()
    return fig


def main():
    # Load model and dataset evaluation metrics
    model, vectorizer = load_production_artifacts()
    metrics, comparison_df, comparison_json = load_metrics_and_comparison()

    # =========================================================================
    # SIDEBAR: System Status, Test Metrics & Quick Controls
    # =========================================================================
    with st.sidebar:
        st.title("🛡️ Moderation System")
        st.caption("AI-Powered Text Content Moderation")

        if model is not None and vectorizer is not None:
            st.success("🟢 **Model:** Loaded & Ready")
            st.markdown(
                f"- **Algorithm:** `{metrics.get('selected_model', 'Logistic Regression')}`\n"
                f"- **Features:** `25,000 TF-IDF n-grams`\n"
                f"- **Dataset:** `Jigsaw / Wikipedia (159k rows)`"
            )
        else:
            st.error("🔴 **Model Artifacts Missing**")
            st.warning("Please run `python train.py` in your terminal to train and save models.")

        st.divider()

        # Decision threshold slider for viva demonstration
        st.subheader("⚙️ Decision Threshold")
        threshold = st.slider(
            "Toxicity Cutoff",
            min_value=0.20,
            max_value=0.80,
            value=0.50,
            step=0.05,
            help="Scores above this cutoff are classified as Toxic. Lower values increase recall (catch more toxic comments); higher values increase precision (fewer false alarms)."
        )
        if threshold == 0.50:
            st.caption("Standard 50% probability threshold.")
        elif threshold < 0.50:
            st.caption(f"Strict Moderation: flags comments at ≥ {int(threshold*100)}% toxic confidence.")
        else:
            st.caption(f"Lenient Moderation: flags comments only at ≥ {int(threshold*100)}% toxic confidence.")

        st.divider()

        # Quick Performance Metrics Summary
        if metrics:
            st.subheader("🎯 Test Performance")
            st.caption(f"Evaluated on {metrics.get('test_samples', 31915):,} held-out comments")
            col_a, col_b = st.columns(2)
            col_a.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
            col_b.metric("Precision", f"{metrics['precision']*100:.1f}%")

            col_c, col_d = st.columns(2)
            col_c.metric("Recall", f"{metrics['recall']*100:.1f}%")
            col_d.metric("Toxic F1", f"{metrics['f1_score']*100:.1f}%")

        st.divider()
        st.markdown(
            "<small>College ML Project | Pair Programming with Antigravity</small>",
            unsafe_allow_html=True
        )

    # =========================================================================
    # MAIN HEADER & TAB NAVIGATION
    # =========================================================================
    st.title("🛡️ AI Content Moderation System")
    st.markdown(
        "A complete Machine Learning system for real-time online toxicity detection, "
        "featuring an 80:20 stratified benchmark across **3 standard ML algorithms** "
        "on the Wikipedia/Jigsaw toxicity challenge dataset."
    )

    tab_predict, tab_compare, tab_eval, tab_viva, tab_ethics = st.tabs([
        "🔍 Live Content Moderation",
        "📊 3-Model Comparison",
        "📈 Detailed Evaluation",
        "🎓 Viva & Theory Guide",
        "⚖️ Responsible AI"
    ])

    # =========================================================================
    # TAB 1: LIVE CONTENT MODERATION (MAIN PREDICTION INTERFACE)
    # =========================================================================
    with tab_predict:
        st.subheader("💬 Test Live Text Comments")
        st.write(
            "Enter any comment, tweet, or forum message to evaluate whether the text "
            "contains toxic, abusive, or harmful language."
        )

        if model is None or vectorizer is None:
            st.error("⚠️ Model artifacts are not loaded. Run `python train.py` to train the models.")
            return

        # Preset test samples for convenient viva demonstrations
        st.markdown("**Quick Test Samples:**")
        sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)

        selected_sample = None
        if sample_col1.button("🌟 Positive / Constructive"):
            selected_sample = "Thank you so much for this helpful explanation! Keep up the fantastic work."
        if sample_col2.button("💡 Polite Disagreement"):
            selected_sample = "I respectfully disagree with your conclusion. Let's examine the published references."
        if sample_col3.button("🚫 Personal Insult"):
            selected_sample = "You are a complete idiot and you should delete your account."
        if sample_col4.button("🚫 Extreme Toxicity"):
            selected_sample = "Shut up you absolute fool, nobody cares about your pathetic garbage."

        # Input text area
        user_comment = st.text_area(
            label="Comment input",
            value=selected_sample if selected_sample else "",
            placeholder="Type or paste any online comment here to analyze...",
            height=120,
            key="comment_input_box"
        )

        analyze_button = st.button("🔍 Analyze Content", type="primary", use_container_width=True)

        if analyze_button or (user_comment and selected_sample):
            cleaned_input = user_comment.strip()
            if not cleaned_input:
                st.warning("⚠️ Input is empty. Please enter or select a comment to analyze.")
            else:
                # 1. Transform input text with fitted TF-IDF Vectorizer
                input_vec = vectorizer.transform([cleaned_input])

                # 2. Compute probabilities using Logistic Regression
                probabilities = model.predict_proba(input_vec)[0]
                prob_clean = float(probabilities[0])
                prob_toxic = float(probabilities[1])

                # 3. Apply custom threshold
                is_toxic = prob_toxic >= threshold

                st.divider()
                st.subheader("📋 Moderation Verdict")

                card_col, score_col = st.columns([1.1, 1.9])

                with card_col:
                    if is_toxic:
                        st.error("### 🚨 FLAGGED AS TOXIC")
                        st.markdown(f"**Classification:** `TOXIC / HARMFUL`")
                        st.markdown(f"**Action Recommended:** `Auto-block / Flag for Review`")
                        st.markdown(f"**Threshold Applied:** `≥ {int(threshold*100)}%`")
                    else:
                        st.success("### ✅ APPROVED AS SAFE")
                        st.markdown(f"**Classification:** `CLEAN / NON-TOXIC`")
                        st.markdown(f"**Action Recommended:** `Safe to publish`")
                        st.markdown(f"**Threshold Applied:** `< {int(threshold*100)}%`")

                with score_col:
                    st.markdown("##### Model Confidence Probabilities")
                    st.write(f"**Toxicity Probability:** `{prob_toxic * 100:.2f}%`")
                    st.progress(min(prob_toxic, 1.0))

                    st.write(f"**Clean / Safe Probability:** `{prob_clean * 100:.2f}%`")
                    st.progress(min(prob_clean, 1.0))

                # Explainability Section
                contributing_words = extract_word_influences(cleaned_input, vectorizer, model, top_n=6)
                if contributing_words:
                    st.divider()
                    st.subheader("🔎 Explainability: Key Feature Contributions")
                    st.caption(
                        "Words extracted from the comment and their linear log-odds contribution "
                        "in the Logistic Regression model. Higher positive values increase toxicity probability."
                    )

                    cols = st.columns(len(contributing_words))
                    for i, (word, score, tfidf_val) in enumerate(contributing_words):
                        with cols[i]:
                            badge_color = "red" if score > 0 else "green"
                            label = "Toxic" if score > 0 else "Clean"
                            st.metric(
                                label=f"Word: '{word}'",
                                value=f"{score:+.3f}",
                                delta=label,
                                delta_color="inverse" if score > 0 else "normal"
                            )

                    st.info(
                        "ℹ️ **Explainability Note:** These token scores represent statistical correlations learned "
                        "from the 159k training comments. They do not constitute a definitive judgment of human intent."
                    )

    # =========================================================================
    # TAB 2: 3-MODEL COMPARISON BENCHMARK
    # =========================================================================
    with tab_compare:
        st.subheader("📊 3-Algorithm Benchmark Comparison")
        st.write(
            "Every algorithm was trained on the exact same **80% training set (127,656 comments)** "
            "with identical TF-IDF features and evaluated strictly on the **held-out 20% test set (31,915 comments)**."
        )

        if comparison_df is not None:
            # Display Selected Model Highlight
            selected_algo = metrics.get("selected_model", "Logistic Regression") if metrics else "Logistic Regression"
            st.success(
                f"🏆 **Selected Production Model: {selected_algo}**  \n"
                f"Selected for achieving the highest Toxic-Class F1-Score (**76.37%**) with high Precision (**90.18%**) "
                f"and near-instant training time (**0.95s**)."
            )

            # Format comparison table for display
            display_df = comparison_df.copy()
            display_df["Accuracy"] = (display_df["Accuracy"] * 100).map("{:.2f}%".format)
            display_df["Toxic_Precision"] = (display_df["Toxic_Precision"] * 100).map("{:.2f}%".format)
            display_df["Toxic_Recall"] = (display_df["Toxic_Recall"] * 100).map("{:.2f}%".format)
            display_df["Toxic_F1"] = (display_df["Toxic_F1"] * 100).map("{:.2f}%".format)
            display_df["Training_Time_Sec"] = display_df["Training_Time_Sec"].map("{:.2f}s".format)

            display_cols = [
                "Algorithm", "Accuracy", "Toxic_Precision", "Toxic_Recall", "Toxic_F1", "Training_Time_Sec"
            ]
            st.dataframe(
                display_df[display_cols].rename(columns={
                    "Toxic_Precision": "Precision (Toxic)",
                    "Toxic_Recall": "Recall (Toxic)",
                    "Toxic_F1": "F1-Score (Toxic)",
                    "Training_Time_Sec": "Training Time"
                }),
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            # F1-Score Bar Chart
            st.subheader("📈 Toxic-Class F1-Score Comparison")
            st.caption("F1-score is the harmonic mean of Precision and Recall — the primary ranking metric.")

            chart_data = pd.DataFrame({
                "Algorithm": comparison_df["Algorithm"],
                "F1-Score (%)": comparison_df["Toxic_F1"] * 100
            }).set_index("Algorithm")

            st.bar_chart(chart_data)

            # Saved comparison plot image fallback
            if os.path.exists(COMPARISON_CHART_PATH):
                with st.expander("🖼️ View High-Resolution Comparison Plot (Saved by train.py)"):
                    st.image(COMPARISON_CHART_PATH, caption="Model Comparison: Toxic-Class F1-Score on Held-Out Test Set")

            st.divider()

            # Multi-Metric Comparison (Grouped Bar Chart)
            st.subheader("📊 Multi-Metric Comparison (Grouped Bar Chart)")
            st.caption("Side-by-side comparison of Accuracy, Toxic Precision, Toxic Recall, and Toxic F1-Score across all 3 benchmarked models.")
            grouped_fig = render_grouped_comparison_chart(comparison_df)
            st.pyplot(grouped_fig)
            plt.close(grouped_fig)

            st.divider()

            # Viva Insight: The Imbalanced Accuracy Paradox
            st.subheader("💡 Why Accuracy Alone is Insufficient (The Accuracy Paradox)")
            st.markdown("""
            In our dataset, **89.8% of comments are Clean** and only **10.2% are Toxic**.
            - A trivial "dummy classifier" that predicts *Clean* for every comment without even reading the text would score **89.8% Accuracy**, yet be **0% effective** at stopping abusive behavior!
            - Notice that while all three models achieve >94% accuracy, their **Recall ranges from 53% to 68%**.
            - This is why **Toxic Precision**, **Toxic Recall**, and **F1-Score** are the critical metrics for content moderation.
            """)
        else:
            st.info("No model comparison data found. Please run `python train.py` to generate the comparison benchmark.")

    # =========================================================================
    # TAB 3: DETAILED EVALUATION & CONFUSION MATRIX
    # =========================================================================
    with tab_eval:
        st.subheader("📈 Detailed Model Evaluation (Held-Out Test Set)")
        st.write(
            "Rigorous evaluation performed strictly on **31,915 held-out test comments** (20% stratified test split) "
            "that the models never saw during training."
        )

        if metrics and "confusion_matrix" in metrics:
            cm = metrics["confusion_matrix"]
            tn = cm.get("true_negatives", 0)
            fp = cm.get("false_positives", 0)
            fn = cm.get("false_negatives", 0)
            tp = cm.get("true_positives", 0)
            total_test = tn + fp + fn + tp

            # =================================================================
            # 1. METRIC CARDS SUMMARY (Formatted to 2 decimal places)
            # =================================================================
            st.markdown("#### 🎯 Performance Metric Cards")
            eval_col1, eval_col2, eval_col3, eval_col4 = st.columns(4)

            with eval_col1:
                st.metric(
                    label="Overall Accuracy",
                    value=f"{metrics.get('accuracy', 0.9583) * 100:.2f}%"
                )
                st.caption("Percentage of correct predictions across both non-toxic and toxic test comments.")

            with eval_col2:
                st.metric(
                    label="Precision (Toxic Class)",
                    value=f"{metrics.get('precision', 0.9018) * 100:.2f}%"
                )
                st.caption("Positive Class: **Toxic**. Fraction of model-flagged comments that were genuinely toxic (low false alarms).")

            with eval_col3:
                st.metric(
                    label="Recall (Toxic Class)",
                    value=f"{metrics.get('recall', 0.6622) * 100:.2f}%"
                )
                st.caption("Positive Class: **Toxic**. Fraction of all actual toxic comments successfully intercepted by the model.")

            with eval_col4:
                st.metric(
                    label="F1-Score (Toxic Class)",
                    value=f"{metrics.get('f1_score', 0.7637) * 100:.2f}%"
                )
                st.caption("Positive Class: **Toxic**. Harmonic mean balancing precision and recall for the positive class.")

            st.info(
                "📌 **Evaluation Protocol & Positive Class Definition:** The positive class is explicitly defined as **Toxic (Class 1)**. "
                "All metrics, confusion matrix counts, and classification reports below are measured on the **31,915 held-out test comments** "
                f"at the standard model decision threshold of **0.50** (seed = 42). "
                f"*(Note: The interactive sidebar threshold is currently set to **{threshold:.2f}** for live moderation testing)*."
            )

            st.divider()

            # =================================================================
            # 2. CONFUSION MATRIX (Numerical Matrix + Heatmap Visualization)
            # =================================================================
            st.markdown("#### 🔍 Confusion Matrix Analysis")
            st.caption(
                "A 2x2 matrix comparing ground truth actual test labels against model predictions. "
                "Fixed class order: [0, 1] where 0 = Non-toxic and 1 = Toxic."
            )

            cm_left, cm_right = st.columns([1.1, 1.2])

            with cm_left:
                st.markdown("##### 🔢 Numerical Confusion Matrix")
                cm_table = pd.DataFrame(
                    [[f"{tn:,} (TN)", f"{fp:,} (FP)"],
                     [f"{fn:,} (FN)", f"{tp:,} (TP)"]],
                    columns=["Predicted: Non-toxic (0)", "Predicted: Toxic (1)"],
                    index=["Actual: Non-toxic (0)", "Actual: Toxic (1)"]
                )
                st.table(cm_table)
                st.caption(f"Total test instances evaluated: **{total_test:,}** (Sum = TN + FP + FN + TP)")

                st.markdown("**Matrix Terminology Breakdown:**")
                st.markdown(f"- **True Negatives (TN = {tn:,}):** Non-toxic comments correctly approved as safe (**99.2%** accuracy on benign text).")
                st.markdown(f"- **False Positives (FP = {fp:,}):** Non-toxic comments incorrectly flagged as toxic (**only 0.8%** false alarm rate).")
                st.markdown(f"- **False Negatives (FN = {fn:,}):** Toxic comments missed by the model at the standard 0.50 threshold.")
                st.markdown(f"- **True Positives (TP = {tp:,}):** Toxic comments correctly intercepted and blocked.")

            with cm_right:
                st.markdown("##### 🎨 Confusion Matrix Heatmap")
                cm_array = np.array([[tn, fp], [fn, tp]])
                cm_fig = render_confusion_matrix_heatmap(
                    cm_array,
                    class_names=["Non-toxic (0)", "Toxic (1)"]
                )
                st.pyplot(cm_fig)
                plt.close(cm_fig)

            st.divider()

            # =================================================================
            # 3. CLASSIFICATION REPORT (Complete Scikit-Learn Report)
            # =================================================================
            if comparison_json and "detailed_reports" in comparison_json:
                selected_algo = metrics.get("selected_model", "Logistic Regression")
                report = comparison_json["detailed_reports"].get(selected_algo, {}).get("classification_report", {})

                if report:
                    st.markdown(f"#### 📋 Scikit-Learn Classification Report: `{selected_algo}`")
                    st.write(
                        "Detailed per-class precision, recall, and F1-score breakdown, along with macro and weighted averages. "
                        "Calculated on held-out test data using `sklearn.metrics.classification_report(..., zero_division=0)`."
                    )

                    rep_rows = []

                    # Class 0: Non-toxic
                    if "Clean (0)" in report:
                        c = report["Clean (0)"]
                        rep_rows.append({
                            "Class / Category": "Non-toxic (Class 0)",
                            "Precision": f"{c['precision'] * 100:.2f}%",
                            "Recall": f"{c['recall'] * 100:.2f}%",
                            "F1-Score": f"{c['f1-score'] * 100:.2f}%",
                            "Support": f"{int(c['support']):,}"
                        })

                    # Class 1: Toxic
                    if "Toxic (1)" in report:
                        t = report["Toxic (1)"]
                        rep_rows.append({
                            "Class / Category": "Toxic (Class 1 - Positive Class)",
                            "Precision": f"{t['precision'] * 100:.2f}%",
                            "Recall": f"{t['recall'] * 100:.2f}%",
                            "F1-Score": f"{t['f1-score'] * 100:.2f}%",
                            "Support": f"{int(t['support']):,}"
                        })

                    # Accuracy row
                    if "accuracy" in report:
                        acc_val = report["accuracy"]
                        rep_rows.append({
                            "Class / Category": "Accuracy",
                            "Precision": "-",
                            "Recall": "-",
                            "F1-Score": f"{acc_val * 100:.2f}%",
                            "Support": f"{total_test:,}"
                        })

                    # Macro Average
                    if "macro avg" in report:
                        m = report["macro avg"]
                        rep_rows.append({
                            "Class / Category": "Macro Average",
                            "Precision": f"{m['precision'] * 100:.2f}%",
                            "Recall": f"{m['recall'] * 100:.2f}%",
                            "F1-Score": f"{m['f1-score'] * 100:.2f}%",
                            "Support": f"{int(m['support']):,}"
                        })

                    # Weighted Average
                    if "weighted avg" in report:
                        w = report["weighted avg"]
                        rep_rows.append({
                            "Class / Category": "Weighted Average",
                            "Precision": f"{w['precision'] * 100:.2f}%",
                            "Recall": f"{w['recall'] * 100:.2f}%",
                            "F1-Score": f"{w['f1-score'] * 100:.2f}%",
                            "Support": f"{int(w['support']):,}"
                        })

                    rep_df = pd.DataFrame(rep_rows)
                    st.dataframe(rep_df, use_container_width=True, hide_index=True)

                    st.markdown("""
                    > ℹ️ **Understanding Support & Averages:**
                    > - **Support:** The actual number of true test instances for each class in the held-out test set (**28,670 Non-toxic** and **3,245 Toxic** comments, summing to **31,915** total test samples).
                    > - **Macro Average:** Computes the unweighted arithmetic mean of metrics across both classes, treating both classes equally regardless of sample frequency.
                    > - **Weighted Average:** Computes the average weighted by the support of each class, accounting for the natural class distribution.
                    """)

    # =========================================================================
    # TAB 4: VIVA PREPARATION & THEORY GUIDE
    # =========================================================================
    with tab_viva:
        st.subheader("🎓 College Viva Q&A & ML Concepts")
        st.write("Ready answers and key explanations based directly on this project's implementation:")

        with st.expander("1. What problem does this project solve?", expanded=True):
            st.write(
                "It automates the detection of toxic, offensive, and abusive language in user-generated comments. "
                "Manual content moderation cannot scale to millions of comments and takes an emotional toll on human moderators."
            )

        with st.expander("2. Why use TF-IDF instead of simple CountVectorizer?"):
            st.write(
                "CountVectorizer simply counts word frequency, meaning common words like 'the', 'is', 'you' receive highest weights. "
                "TF-IDF (Term Frequency-Inverse Document Frequency) downweights words that appear across all comments and boosts words "
                "that are distinctive to specific contexts (e.g. insults and slurs)."
            )

        with st.expander("3. How do the 3 benchmarked algorithms work?"):
            st.markdown("""
            1. **Logistic Regression:** Computes a weighted linear sum of TF-IDF features and passes it through a sigmoid function to output calibrated probabilities (0.0 to 1.0). Best suited for high-dimensional, sparse text.
            2. **Decision Tree:** Splits data recursively using feature thresholds based on Gini Impurity or Information Gain. Struggles with text because each split only evaluates one word at a time.
            3. **Random Forest:** An ensemble of multiple decision trees trained on bootstrap data samples with random feature subsets. High precision and recall, but higher training time and memory footprint.
            """)

        with st.expander("4. Why was Logistic Regression selected over Random Forest and Decision Tree?"):
            st.write(
                "Logistic Regression achieved the highest Toxic-Class F1-score (76.37%) and highest Precision (90.18%) "
                "while training in under 1 second. TF-IDF feature spaces are high-dimensional (25,000 features) and linearly separable, "
                "which matches Logistic Regression's mathematical strengths perfectly."
            )

        with st.expander("5. What is the difference between Precision and Recall?"):
            st.markdown("""
            - **Precision:** Out of all comments the model *flagged as toxic*, how many were *actually toxic*? (High precision = few false alarms).
            - **Recall:** Out of all *actually toxic comments*, how many did the model *catch*? (High recall = few missed toxic comments).
            - **F1-Score:** The harmonic mean: `2 * (Precision * Recall) / (Precision + Recall)`.
            """)

    # =========================================================================
    # TAB 5: RESPONSIBLE AI & ETHICAL CONSIDERATIONS
    # =========================================================================
    with tab_ethics:
        st.subheader("⚖️ Responsible AI & Ethical Considerations")
        st.markdown("""
        Deploying machine learning models for automated content moderation involves critical societal and ethical responsibilities:

        1. **Imperfection of Automated Classifiers:**
           No statistical model achieves 100% precision or recall. Predictions should serve as a **decision-support tool** for human moderators rather than an unquestionable authority.

        2. **Dataset Bias & Cultural Nuance:**
           Training datasets often reflect demographic biases present in online forums. Certain colloquial terms, dialectic expressions, or reclaimed language may be falsely flagged by models trained on generic web corpora.

        3. **Context and Sarcasm:**
           TF-IDF n-gram representations evaluate word frequencies and pairs, but lack deep syntactic understanding of sarcasm, satire, or situational context.

        4. **Human-in-the-Loop Architecture:**
           In production systems, borderline comments (e.g. toxicity scores between 40% and 70%) should be queued for human review rather than permanently banned automatically.
        """)


if __name__ == "__main__":
    main()
