# 🛡️ AI Content Moderation System
### Real-Time Text Toxicity Classification & 3-Model ML Benchmark

A complete, production-ready Machine Learning system engineered to detect toxic, abusive, and harmful comments in online platforms. Built with an end-to-end NLP pipeline, an **80:20 stratified benchmark comparing 3 standard classification algorithms**, genuine held-out test evaluation, and an interactive **Streamlit** dashboard.

---

## 📌 1. Project Overview & Problem Statement
User-generated content drives modern social networks, news portals, discussion forums, and collaborative workspaces. However, online toxicity, harassment, hate speech, and profanity degrade user experience, drive away community members, and can lead to severe legal and emotional consequences.

Manual moderation of millions of daily comments is economically unviable and inflicts immense psychological distress on human reviewers. This project delivers an automated **AI Content Moderation System** that ingests raw user comments, extracts numerical features using **TF-IDF**, and classifies comments in real time with probabilistic confidence scores and transparent explainability.

---

## ✨ 2. Key Features
- **Real Benchmark Dataset:** Trained on **159,571 human-annotated comments** from the official Wikipedia / Jigsaw challenge.
- **3-Algorithm Comparative Benchmark:** Rigorous, unbiased comparison of:
  1. *Logistic Regression* (Selected Production Model — **76.37% Toxic F1**)
  2. *Decision Tree Classifier*
  3. *Random Forest Classifier*
- **Strict Adherence to Guidelines:** Zero Support Vector Machines (no SVM/SVC) per project requirements.
- **Genuine Held-Out Evaluation:** 31,915 test samples (20%) reserved strictly for evaluation; zero data leakage or fabricated scores.
- **Real-Time Interactive Dashboard:** Built with Streamlit, offering:
  - Instant text toxicity analysis with clean/toxic verdicts.
  - Granular probability progress meters.
  - Adjustable decision threshold slider (0.20 to 0.80) to demonstrate Precision-Recall trade-offs.
  - **Explainable AI (XAI):** Linear feature-contribution breakdown showing which words triggered the verdict.
  - Full 3-model comparison tables and interactive F1-score charts.
  - Detailed confusion matrix breakdown and classification reports.
  - Built-in College Viva Preparation and Responsible AI ethics guides.
- **Automated Verification Suite:** Dedicated test suite (`test_project.py`) verifying dataset integrity, artifact loading, inference, explainability, and metrics across all 3 models.

---

## 💻 3. Tech Stack
- **Programming Language:** Python 3
- **Data Engineering:** `pandas`, `numpy`
- **Machine Learning & NLP:** `scikit-learn` (`TfidfVectorizer`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `train_test_split`, `metrics`)
- **Model Serialization:** `joblib`
- **Interactive UI & Dashboard:** `streamlit`
- **Data Visualization:** `matplotlib`, `altair`

---

## 📊 4. Dataset Source & Target Definition
- **Source:** Official Kaggle / Wikipedia **Jigsaw Toxic Comment Classification Challenge** dataset.
- **Total Records:** 159,571 comments.
- **Target Label Definition:**  
  The Jigsaw dataset provides multi-label annotations across six toxicity categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.  
  For binary content moderation:
  $$\text{Target} = \begin{cases} 1 \text{ (Toxic)}, & \text{if any of the 6 categories is flagged} \\ 0 \text{ (Clean)}, & \text{otherwise} \end{cases}$$
- **Class Breakdown:**
  - **Clean Comments:** 143,346 (89.8%)
  - **Toxic Comments:** 16,225 (10.2%)
- **Data Partitioning:** Stratified 80:20 split (`random_state=42`):
  - **Training Set:** 127,656 comments (Clean: 114,676, Toxic: 12,980)
  - **Testing Set:** 31,915 comments (Clean: 28,670, Toxic: 3,245)

---

## ⚙️ 5. Machine Learning Pipeline

```
Raw User Text
      │
      ▼
Text Preprocessing
(Lowercasing, URL removal, Non-alphabetic cleanup, Whitespace normalization)
      │
      ▼
TF-IDF Vectorization (Unigrams + Bigrams, 25,000 top features, Sublinear TF)
[Fitted STRICTLY on Training Data Only]
      │
      ▼
3-Model Comparative Benchmark
├── 1. Logistic Regression (liblinear, C=3.0) ──> [Selected Production Model]
├── 2. Decision Tree (max_depth=30)
└── 3. Random Forest (40 trees, min_samples_split=10)
      │
      ▼
Held-Out Test Evaluation (31,915 Unseen Comments)
      │
      ▼
Serialization & Real-Time Deployment (joblib + Streamlit)
```

---

## 🧠 6. Feature Extraction: Why TF-IDF?
Raw text cannot be processed directly by statistical models.
- **CountVectorizer (Bag of Words)** merely counts word frequencies. Under this representation, high-frequency stop words like *"the"*, *"is"*, *"and"* dominate feature weights despite conveying zero toxicity.
- **TF-IDF (Term Frequency-Inverse Document Frequency)** overcomes this:
  $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$
  - **Term Frequency ($\text{TF}$):** Measures how often token $t$ appears in comment $d$. Sublinear scaling ($1 + \log(\text{TF})$) dampens repeated words.
  - **Inverse Document Frequency ($\text{IDF}$):** Heavily penalizes words appearing across many documents, while dramatically boosting distinctive words (such as targeted insults and slurs) that appear in localized toxic contexts.
  - **N-Grams (1, 2):** Captures single words (*"idiot"*) and two-word combinations (*"shut up"*, *"go away"*).

---

## 🔬 7. Machine Learning Algorithms Compared

| Algorithm | Theoretical Working Principle | Key Strengths in Text NLP |
| :--- | :--- | :--- |
| **1. Logistic Regression** | Fits a linear decision hyperplane $z = \mathbf{w}^T \mathbf{x} + b$ and maps it to a probability $[0, 1]$ via the sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$. Regularized with L2 penalty. | Exceptional with high-dimensional sparse TF-IDF matrices; fast convergence; well-calibrated probabilities. |
| **2. Decision Tree** | Recursively partitions the feature space using Gini Impurity or Entropy threshold splits. | Highly interpretable single tree; however, text relies on combinations of words which single-variable splits fail to capture well. |
| **3. Random Forest** | An ensemble of $N$ de-correlated decision trees trained via bootstrap aggregation (bagging) with random feature subspaces. | Reduces variance; captures non-linear interactions; higher computational and memory overhead. |

---

## 📈 8. Actual Model Comparison Results (Held-Out Test Set)

Evaluated strictly on **31,915 held-out test comments** (20% test partition):

| Algorithm | Accuracy | Precision (Toxic) | Recall (Toxic) | F1-Score (Toxic) | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Selected)** | **95.83%** | **90.18%** | **66.22%** | **76.37%** | **0.94s** |
| **Random Forest** | 95.59% | 86.07% | 67.58% | 75.71% | 23.01s |
| **Decision Tree** | 94.47% | 87.71% | 53.00% | 66.08% | 8.03s |

### 🏆 Preferred Model Selection Rationale
**Logistic Regression** is selected as the production model because:
1. **Highest Toxic F1-Score (76.37%):** Effectively balances false alarms and missed toxic comments.
2. **Highest Precision (90.18%):** Out of all comments flagged as toxic, over 90% are truly toxic, preventing unfair moderation of benign users.
3. **Training Efficiency (0.94s):** 24x faster than Random Forest (23.01s) with superior F1.
4. **Lightweight Deployment:** Model artifact is only ~200 KB, enabling sub-millisecond inference on low-cost infrastructure.

---

## 🔍 9. Confusion Matrix Interpretation (Production Model)

```text
                              PREDICTED
                      Clean (0)       Toxic (1)
ACTUAL   Clean (0)   28,436 (TN)       234 (FP)
         Toxic (1)    1,096 (FN)     2,149 (TP)
```

- **True Negatives (TN = 28,436):** Safe comments correctly classified as Clean (**99.2% accuracy on non-toxic text**).
- **False Positives (FP = 234):** Safe comments incorrectly flagged as toxic (**only 0.8% false alarm rate**).
- **False Negatives (FN = 1,096):** Toxic comments missed by the default 50% cutoff. Platforms needing stricter enforcement can lower the threshold to 0.35 in the dashboard.
- **True Positives (TP = 2,149):** Toxic comments correctly intercepted and blocked.

---

## 📁 10. Project Folder Structure
```text
ML project/
│
├── data/
│   └── train.csv                    # Real Jigsaw Toxic Comment Dataset (159k rows, 68.8 MB)
│
├── models/
│   ├── toxic_model.joblib            # Serialized Logistic Regression production classifier
│   ├── tfidf_vectorizer.joblib       # Serialized TF-IDF Vectorizer (25,000 features)
│   ├── metrics.json                 # Genuine evaluation metrics (JSON)
│   ├── model_comparison.csv         # 3-algorithm benchmark summary table (CSV)
│   ├── model_comparison.json        # Detailed classification reports & confusion matrices
│   └── f1_score_comparison.png      # High-resolution benchmark F1 bar chart
│
├── app.py                           # Interactive Streamlit Web Dashboard
├── train.py                         # Complete 3-model training, evaluation & benchmark script
├── download_data.py                 # Automated dataset downloader utility
├── test_project.py                  # Automated test suite (8 verification suites)
├── VIVA_PREPARATION.md              # Complete 19-question College Viva & Interview Guide
├── requirements.txt                 # Project dependencies
├── .gitignore                       # Clean Git ignore configuration
└── README.md                        # Master project documentation
```

---

## 🚀 11. Installation & Execution Guide

### 1. Clone or Open Project Directory
```bash
cd "ML project"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
Verify dataset, models, inference, and metrics:
```bash
python test_project.py
```

### 4. Run Model Training & Benchmark
Trains all 3 models, generates metrics, comparison tables, and plots:
```bash
python train.py
```

### 5. Launch the Streamlit Dashboard
```bash
python -m streamlit run app.py
```
Open your browser and navigate to:  
👉 **`http://localhost:8501`**

---

## 📸 12. Application Screenshots

| Section | Description |
| :--- | :--- |
| **Live Moderation Interface** | *Text area with sample prompts, real-time prediction card, toxicity meters, and XAI keyword breakdown.* |
| **Model Comparison Tab** | *Comparative benchmark table across all 3 algorithms with interactive F1-score bar charts.* |
| **Detailed Evaluation Tab** | *Held-out test set confusion matrix cards, matrix table, and per-class classification reports.* |
| **College Viva Guide Tab** | *Integrated 19-question revision guide directly accessible in the web UI.* |

*(Screenshots can be saved directly into an `assets/` folder for college submission reports).*

---

## ⚖️ 13. Limitations, Dataset Bias & Responsible AI
- **Decision-Support Tool:** Automated classifiers should aid human moderators, not serve as final arbiters for permanent bans.
- **Context & Sarcasm:** Statistical n-gram models cannot fully understand sarcasm, self-deprecation, or dry humor.
- **Dataset Bias:** Training comments originate from Wikipedia talk pages; colloquial terms or African-American Vernacular English (AAVE) may be disproportionately flagged if unmonitored.
- **Explainability Transparency:** The feature-importance scores reflect statistical associations in the training set and do not constitute a definitive judgment of human intent.
