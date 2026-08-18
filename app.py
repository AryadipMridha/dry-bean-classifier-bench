"""
Dry Bean Classifier Bench - ML Assignment 2 (BITS ID 2025AC05211)

Streamlit front-end for five pre-trained scikit-learn classifiers on the
UCI Dry Bean dataset. Upload the test CSV, pick a model, read the metrics.

Local run:  streamlit run app.py
"""

import io
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
TARGET = "Class"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}

st.set_page_config(
    page_title="Dry Bean Classifier Bench",
    page_icon=":seedling:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .bean-head {
        background: linear-gradient(90deg, #6b4423 0%, #a9743f 55%, #d9b26a 100%);
        padding: 1.1rem 1.4rem; border-radius: 10px; margin-bottom: 1rem;
      }
      .bean-head h1 { color: #fff8ec; margin: 0; font-size: 1.7rem; }
      .bean-head p  { color: #f2e3c8; margin: .25rem 0 0; font-size: .92rem; }
      div[data-testid="stMetricValue"] { font-size: 1.45rem; }
      .stDataFrame { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# loading helpers
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_bundle():
    models = {}
    for label, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[label] = joblib.load(path)
    encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    return models, encoder, meta


@st.cache_data(show_spinner=False)
def read_csv(raw_bytes):
    return pd.read_csv(io.BytesIO(raw_bytes))


def evaluate(y_true, y_pred, y_proba, n_classes):
    """Macro averaging: the seven bean classes are imbalanced, so every class
    should weigh the same regardless of how many samples it contributes."""
    if n_classes == 2:
        auc_value = roc_auc_score(y_true, y_proba[:, 1])
    else:
        auc_value = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc_value,
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def draw_confusion(y_true, y_pred, class_names, normalise):
    cm = confusion_matrix(y_true, y_pred)
    fmt = "d"
    if normalise:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="YlOrBr",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted bean type")
    ax.set_ylabel("Actual bean type")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    return fig


def draw_roc(y_true, y_proba, class_names):
    y_bin = label_binarize(y_true, classes=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for idx, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_proba[:, idx])
        ax.plot(fpr, tpr, lw=1.6, label=f"{name} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.9)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-Rest ROC per bean class")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# sidebar - data source and model choice
# ----------------------------------------------------------------------------
models, encoder, meta = load_bundle()
class_names = list(encoder.classes_)
feature_names = meta["feature_names"]

st.markdown(
    """
    <div class="bean-head">
      <h1>Dry Bean Classifier Bench</h1>
      <p>ML Assignment 2 &middot; BITS ID 2025AC05211 &middot; UCI Dry Bean dataset
      (13,611 samples &middot; 16 morphological features &middot; 7 bean varieties)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("1. Test data")
    uploaded = st.file_uploader(
        "Upload test CSV",
        type=["csv"],
        help="Use test_data.csv from the repo, or any CSV with the same 16 feature "
        "columns. A 'Class' column is optional - without it the app only predicts.",
    )
    # Fallback only - an uploaded file always wins, so this checkbox can never
    # go stale and swallow a file the user just dropped in.
    use_bundled = st.checkbox(
        "Fall back to bundled test_data.csv when nothing is uploaded", value=True
    )

    st.header("2. Model")
    chosen = st.selectbox("Classifier", list(models.keys()), index=4)

    st.header("3. Display")
    normalise_cm = st.toggle("Normalise confusion matrix", value=False)
    show_roc = st.toggle("Show per-class ROC curves", value=True)

    st.divider()
    st.caption(
        f"**Dataset:** {meta['dataset']}  \n"
        f"**Split:** {int((1 - meta['test_size']) * 100)}/"
        f"{int(meta['test_size'] * 100)} stratified, seed {meta['random_state']}  \n"
        f"**Averaging:** {meta['averaging']}"
    )

# resolve the dataframe to work with
frame = None
source_note = ""
if uploaded is not None:
    frame = read_csv(uploaded.getvalue())
    source_note = f"uploaded file `{uploaded.name}`"
elif use_bundled and os.path.exists(os.path.join(HERE, "test_data.csv")):
    frame = pd.read_csv(os.path.join(HERE, "test_data.csv"))
    source_note = "bundled `test_data.csv`"

if frame is None:
    st.info("Upload a test CSV from the sidebar, or re-enable the bundled-data fallback.")
    st.stop()

missing = [c for c in feature_names if c not in frame.columns]
if missing:
    st.error(f"CSV is missing {len(missing)} required feature column(s): {missing}")
    st.stop()

X = frame[feature_names]
has_labels = TARGET in frame.columns
st.success(f"Loaded {len(frame):,} rows from {source_note}. Labels present: {has_labels}.")

model = models[chosen]
y_pred_enc = model.predict(X)
y_proba = model.predict_proba(X)
y_pred = encoder.inverse_transform(y_pred_enc)

if has_labels:
    y_true_enc = encoder.transform(frame[TARGET])

tab_eval, tab_compare, tab_preds, tab_data = st.tabs(
    ["Evaluation", "Model comparison", "Predictions", "Dataset"]
)

# ----------------------------------------------------------------------------
with tab_eval:
    st.subheader(f"{chosen} on this test set")

    if not has_labels:
        st.warning(
            "No `Class` column in the uploaded file, so metrics cannot be computed. "
            "See the **Predictions** tab for the model output."
        )
    else:
        scores = evaluate(y_true_enc, y_pred_enc, y_proba, len(class_names))
        cols = st.columns(6)
        for col, (key, val) in zip(cols, scores.items()):
            col.metric(key, f"{val:.4f}")

        left, right = st.columns(2)
        with left:
            st.markdown("**Confusion matrix**")
            st.pyplot(draw_confusion(y_true_enc, y_pred_enc, class_names, normalise_cm))
        with right:
            st.markdown("**Classification report**")
            report = classification_report(
                y_true_enc,
                y_pred_enc,
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )
            st.dataframe(
                pd.DataFrame(report).T.round(4), width="stretch", height=320
            )

        if show_roc:
            st.markdown("**ROC curves (One-vs-Rest)**")
            st.pyplot(draw_roc(y_true_enc, y_proba, class_names))

# ----------------------------------------------------------------------------
with tab_compare:
    st.subheader("All five models on this test set")
    if not has_labels:
        st.warning("Upload a CSV containing the `Class` column to compare models.")
    else:
        rows = {}
        progress = st.progress(0.0, text="Scoring models...")
        for i, (label, pipe) in enumerate(models.items(), start=1):
            pred = pipe.predict(X)
            proba = pipe.predict_proba(X)
            rows[label] = evaluate(y_true_enc, pred, proba, len(class_names))
            progress.progress(i / len(models), text=f"Scored {label}")
        progress.empty()

        table = pd.DataFrame(rows).T[
            ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
        ]
        st.dataframe(
            table.style.format("{:.4f}").highlight_max(axis=0, color="#e8d3a2"),
            width="stretch",
        )

        winner = table["MCC"].idxmax()
        st.success(
            f"Best model by MCC on this test set: **{winner}** "
            f"(MCC {table.loc[winner, 'MCC']:.4f}, accuracy {table.loc[winner, 'Accuracy']:.4f})"
        )

        melted = table.reset_index().melt(
            id_vars="index", var_name="Metric", value_name="Score"
        )
        fig, ax = plt.subplots(figsize=(9, 3.6))
        sns.barplot(data=melted, x="Metric", y="Score", hue="index", palette="YlOrBr", ax=ax)
        ax.set_ylim(0.80, 1.0)
        ax.set_xlabel("")
        ax.legend(title="", fontsize=8, ncols=5, loc="lower center")
        fig.tight_layout()
        st.pyplot(fig)

        st.download_button(
            "Download comparison table (CSV)",
            table.to_csv().encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv",
        )

# ----------------------------------------------------------------------------
with tab_preds:
    st.subheader(f"Row-level predictions - {chosen}")
    out = frame.copy()
    out["Predicted"] = y_pred
    out["Confidence"] = y_proba.max(axis=1).round(4)
    if has_labels:
        out["Correct"] = out[TARGET] == out["Predicted"]
        wrong_only = st.checkbox("Show misclassified rows only", value=False)
        if wrong_only:
            out = out[~out["Correct"]]
        st.caption(f"{len(out):,} rows shown.")
    st.dataframe(out.head(500), width="stretch", height=380)
    st.download_button(
        "Download all predictions (CSV)",
        out.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{chosen.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

# ----------------------------------------------------------------------------
with tab_data:
    st.subheader("Dataset at a glance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows in full dataset", f"{meta['n_rows']:,}")
    c2.metric("Features", meta["n_features"])
    c3.metric("Classes", len(class_names))

    st.markdown("**Class balance in the full dataset**")
    counts = pd.Series(meta["class_counts"]).sort_values(ascending=False)
    st.bar_chart(counts)

    st.markdown("**Feature summary for the loaded test set**")
    st.dataframe(X.describe().T.round(3), width="stretch", height=340)

    st.caption(f"Source: {meta['source']}")
