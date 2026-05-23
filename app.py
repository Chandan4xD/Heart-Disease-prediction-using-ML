# ─────────────────────────────────────────────────────────────────────────────
#  Heart Disease Prediction System
#  Dataset : Cardiovascular Disease Dataset (Indian Population)
#  Department  : Computer Science & Engineering (CSE)
#  College     : BACET
#  Project     : Final Year Project
#  Developed by: Arpan Das, Chandan Kumar Mishra, MD Belal
# ─────────────────────────────────────────────────────────────────────────────

import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix,
    f1_score, precision_score, recall_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor | BACET CSE",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* overall spacing */
  .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }

  /* page header */
  .page-header { text-align:center; margin-bottom:0.5rem; }
  .page-header h1 { font-size:2rem; font-weight:700; color:#b03a2e; margin-bottom:0; }
  .page-header .subtitle { font-size:0.85rem; color:#777; margin-top:0.2rem; }

  /* section labels */
  .section-label {
    font-size:0.72rem; font-weight:600; text-transform:uppercase;
    letter-spacing:0.08em; color:#888; margin-bottom:0.4rem;
  }

  /* parameter card */
  .param-card {
    background:#f9f9f9; border:1px solid #e8e8e8;
    border-radius:10px; padding:1rem 1.2rem; height:100%;
  }

  /* result boxes */
  .result-box {
    border-radius:10px; padding:1rem 1.5rem; margin-top:1rem;
    font-size:1rem; font-weight:600; line-height:1.7;
  }
  .result-positive {
    background:#fdf2f2; border-left:5px solid #b03a2e; color:#7b241c;
  }
  .result-negative {
    background:#eafaf1; border-left:5px solid #1e8449; color:#1a5e36;
  }

  /* metric cards on About tab */
  .kpi { text-align:center; }
  .kpi .val { font-size:1.9rem; font-weight:700; color:#b03a2e; }
  .kpi .lbl { font-size:0.78rem; color:#888; text-transform:uppercase; }

  /* disclaimer box */
  .disclaimer {
    background:#fffbeb; border:1px solid #f0d080; border-radius:8px;
    padding:0.7rem 1rem; font-size:0.82rem; color:#7d6608; margin-top:0.8rem;
  }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & MODEL TRAINING
#  We check for pre-saved artifacts (best_model.pkl etc.) first so the app
#  is fast after the first run.  If they're missing we train fresh.
# ═════════════════════════════════════════════════════════════════════════════

DATASET = "heart.csv"  # Cardiovascular Disease Dataset (Indian Population)
ARTIFACTS = ["best_model.pkl", "scaler.pkl", "results.pkl"]

@st.cache_resource(show_spinner="Preparing models — please wait...")
def load_everything():
    # ── try pre-saved artifacts (validate they match current dataset) ─────────
    if all(os.path.exists(p) for p in ARTIFACTS) and os.path.exists(DATASET):
        df_check    = _load_df()
        feat_check  = [c for c in df_check.columns if c not in ("patientid", "target")]
        with open("scaler.pkl", "rb") as f:
            scaler_check = pickle.load(f)
        if scaler_check.n_features_in_ == len(feat_check):
            # artifacts match the current dataset — load them
            with open("best_model.pkl", "rb") as f: model   = pickle.load(f)
            with open("results.pkl",    "rb") as f: results = pickle.load(f)
            return model, scaler_check, results, {}, feat_check, df_check
        else:
            # stale artifacts from a different dataset — delete and retrain
            for p in ARTIFACTS:
                if os.path.exists(p):
                    os.remove(p)

    # ── fresh training ───────────────────────────────────────────────────────
    if not os.path.exists(DATASET):
        st.error(f"❌ Dataset '{DATASET}' not found. Place it in the same folder as app.py.")
        st.stop()

    df = _load_df()
    feature_cols = [c for c in df.columns if c not in ("patientid", "target")]
    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model_zoo = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=150, random_state=42),
        "SVM":                 SVC(kernel="rbf", probability=True, random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=7),
        "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
    }

    results        = {}
    trained_models = {}
    extra          = {}   # confusion matrices, ROC data per model

    for name, clf in model_zoo.items():
        clf.fit(X_train_sc, y_train)
        preds = clf.predict(X_test_sc)
        proba = clf.predict_proba(X_test_sc)[:, 1] if hasattr(clf, "predict_proba") else None

        results[name] = {
            "Accuracy":  round(accuracy_score (y_test, preds) * 100, 2),
            "Precision": round(precision_score(y_test, preds) * 100, 2),
            "Recall":    round(recall_score   (y_test, preds) * 100, 2),
            "F1-Score":  round(f1_score       (y_test, preds) * 100, 2),
        }
        trained_models[name] = clf

        cm = confusion_matrix(y_test, preds)
        if proba is not None:
            fpr, tpr, _ = roc_curve(y_test, proba)
            roc_auc = auc(fpr, tpr)
        else:
            fpr = tpr = None; roc_auc = None

        extra[name] = {"cm": cm, "fpr": fpr, "tpr": tpr, "auc": roc_auc}

    best_name  = max(results, key=lambda k: results[k]["Accuracy"])
    best_model = trained_models[best_name]

    # save artifacts
    with open("best_model.pkl", "wb") as f: pickle.dump(best_model, f)
    with open("scaler.pkl",     "wb") as f: pickle.dump(scaler, f)
    with open("results.pkl",    "wb") as f: pickle.dump(results, f)

    return best_model, scaler, results, extra, feature_cols, df


def _load_df():
    """Load and lightly clean the dataset."""
    df = pd.read_csv(DATASET)
    # standardise target column name (some versions use 'output')
    if "target" not in df.columns and "output" in df.columns:
        df.rename(columns={"output": "target"}, inplace=True)
    # drop patient ID — not a feature
    if "patientid" in df.columns:
        df.drop(columns=["patientid"], inplace=True, errors="ignore")
    return df


model, scaler, results, extra, feature_cols, df_cached = load_everything()
best_algo = max(results, key=lambda k: results[k]["Accuracy"])


# ═════════════════════════════════════════════════════════════════════════════
#  HELPER : draw a clean bar chart
# ═════════════════════════════════════════════════════════════════════════════

def bar_chart(labels, values, highlight_label=None, xlabel="", title="", color_base="#aed6f1", color_hi="#b03a2e"):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    colors = [color_hi if l == highlight_label else color_base for l in labels]
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.52)
    for bar, val in zip(bars, values):
        ax.text(val + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9.5, color="#333")
    ax.set_xlim(0, 112)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=10.5, pad=8, color="#333")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False, labelsize=9)
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE HEADER
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="page-header">
  <h1>🫀 Heart Disease Prediction System</h1>
  <div class="subtitle">
    Cardiovascular Disease Dataset · Indian Population · 1000 patients · 5-Model Comparison<br>
    Developed by Arpan Das, Chandan Kumar Mishra &amp; MD Belal &nbsp;|&nbsp;
    Dept. of CSE &nbsp;·&nbsp; BACET &nbsp;·&nbsp; Final Year Project
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
#  TABS
# ═════════════════════════════════════════════════════════════════════════════

tab_predict, tab_perf, tab_insights, tab_about = st.tabs([
    "🔍  Predict", "📊  Performance", "🧬  Data Insights", "ℹ️  About"
])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — PREDICT
# ─────────────────────────────────────────────────────────────────────────────
with tab_predict:
    st.markdown("#### Enter Patient's Clinical Details")
    st.caption("All ranges are calibrated for the Indian population dataset used in training.")

    col_a, col_b, col_c = st.columns(3, gap="medium")

    with col_a:
        st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)
        age    = st.slider("Age (years)", 20, 80, 45)
        gender = st.radio("Gender", ["Female", "Male"], index=1, horizontal=True)
        gender_val = 1 if gender == "Male" else 0

        st.markdown('<div class="section-label" style="margin-top:1rem">Chest Pain</div>', unsafe_allow_html=True)
        chestpain = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 – No / Asymptomatic",
                1: "1 – Atypical angina",
                2: "2 – Non-anginal pain",
                3: "3 – Typical angina",
            }[x],
        )

        st.markdown('<div class="section-label" style="margin-top:1rem">ECG</div>', unsafe_allow_html=True)
        restingrelectro = st.selectbox(
            "Resting Electrocardiogram",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "0 – Normal",
                1: "1 – ST-T wave abnormality",
                2: "2 – Left ventricular hypertrophy",
            }[x],
        )

    with col_b:
        st.markdown('<div class="section-label">Blood Pressure & Cholesterol</div>', unsafe_allow_html=True)

        # Indian dataset has higher typical BP (mean ~152)
        restingBP = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=90, max_value=200, value=140,
            help="Normal range for Indians: 90–140 mm Hg"
        )
        serumcholestrol = st.number_input(
            "Serum Cholesterol (mg/dl)",
            min_value=0, max_value=610, value=250,
            help="Indian avg ~311 mg/dl. Enter 0 if not available."
        )

        st.markdown('<div class="section-label" style="margin-top:1rem">Blood Sugar</div>', unsafe_allow_html=True)
        fastingbloodsugar = st.radio(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[0, 1],
            format_func=lambda x: "Yes" if x else "No",
            horizontal=True,
            help="Diabetes is a major risk factor in Indian patients"
        )

        st.markdown('<div class="section-label" style="margin-top:1rem">Heart Rate</div>', unsafe_allow_html=True)
        maxheartrate = st.number_input(
            "Maximum Heart Rate Achieved",
            min_value=60, max_value=220, value=150,
        )

    with col_c:
        st.markdown('<div class="section-label">Exercise Test Results</div>', unsafe_allow_html=True)
        exerciseangia = st.radio(
            "Exercise-Induced Angina",
            options=[0, 1],
            format_func=lambda x: "Yes" if x else "No",
            horizontal=True,
        )
        oldpeak = st.number_input(
            "ST Depression (oldpeak)",
            min_value=0.0, max_value=6.5, value=2.0, step=0.1,
            help="ST depression induced by exercise relative to rest"
        )

        st.markdown('<div class="section-label" style="margin-top:1rem">ST Slope</div>', unsafe_allow_html=True)
        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 – Upsloping",
                1: "1 – Flat",
                2: "2 – Downsloping",
                3: "3 – Steep downslope",
            }[x],
        )

        st.markdown('<div class="section-label" style="margin-top:1rem">Angiography</div>', unsafe_allow_html=True)
        noofmajorvessels = st.selectbox(
            "No. of Major Vessels Coloured by Fluoroscopy",
            options=[0, 1, 2, 3],
            help="0 = best, 3 = most vessels blocked"
        )

    st.markdown("")
    btn_col, _ = st.columns([1, 4])
    with btn_col:
        predict_clicked = st.button("🔮 Predict Result", type="primary", use_container_width=True)

    if predict_clicked:
        # build input as a DataFrame so sklearn gets correct feature names —
        # this prevents the n_features mismatch ValueError entirely
        input_dict = {
            "age":               [age],
            "gender":            [gender_val],
            "chestpain":         [chestpain],
            "restingBP":         [restingBP],
            "serumcholestrol":   [serumcholestrol],
            "fastingbloodsugar": [fastingbloodsugar],
            "restingrelectro":   [restingrelectro],
            "maxheartrate":      [maxheartrate],
            "exerciseangia":     [exerciseangia],
            "oldpeak":           [oldpeak],
            "slope":             [slope],
            "noofmajorvessels":  [noofmajorvessels],
        }
        user_df    = pd.DataFrame(input_dict)[feature_cols]  # enforce trained column order
        scaled     = scaler.transform(user_df)
        prediction = model.predict(scaled)[0]
        confidence = model.predict_proba(scaled)[0] if hasattr(model, "predict_proba") else None

        st.markdown("---")

        if prediction == 1:
            conf_str = f"&nbsp;·&nbsp; Confidence: <b>{confidence[1]*100:.1f}%</b>" if confidence is not None else ""
            st.markdown(f"""
            <div class="result-box result-positive">
              ❤️‍🔥 &nbsp;Heart Disease <u>Detected</u>{conf_str}<br>
              <span style="font-weight:400; font-size:0.88rem">
                The model found patterns consistent with cardiovascular disease.
                Please consult a cardiologist for a formal diagnosis.
              </span>
            </div>""", unsafe_allow_html=True)
        else:
            conf_str = f"&nbsp;·&nbsp; Confidence: <b>{confidence[0]*100:.1f}%</b>" if confidence is not None else ""
            st.markdown(f"""
            <div class="result-box result-negative">
              ✅ &nbsp;No Heart Disease Detected{conf_str}<br>
              <span style="font-weight:400; font-size:0.88rem">
                The model found no significant indicators of cardiovascular disease.
                Regular annual check-ups are still recommended.
              </span>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="disclaimer">
          ⚠️ <b>Medical Disclaimer:</b> This tool is built for academic research purposes only
          and is <b>not</b> a substitute for professional clinical diagnosis.
          Always consult a qualified cardiologist for health decisions.<br>
          <span style="color:#aaa">Model used: <b>{best_algo}</b>
          &nbsp;·&nbsp; Test Accuracy: <b>{results[best_algo]['Accuracy']}%</b></span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab_perf:
    st.markdown("#### Model Comparison")

    df_res = pd.DataFrame(results).T.reset_index()
    df_res.columns = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]

    def _highlight_best(row):
        style = "background-color:#eafaf1; font-weight:600" if row["Model"] == best_algo else ""
        return [style] * len(row)

    st.dataframe(
        df_res.style
              .apply(_highlight_best, axis=1)
              .highlight_max(subset=["Accuracy","Precision","Recall","F1-Score"],
                             color="#d5f5e3", axis=0),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"⭐ Best model by accuracy: **{best_algo}** — {results[best_algo]['Accuracy']}%")

    st.markdown("#### Accuracy Comparison")
    model_names = list(results.keys())
    accs = [results[m]["Accuracy"] for m in model_names]
    st.pyplot(bar_chart(model_names, accs, highlight_label=best_algo,
                        xlabel="Accuracy (%)", title=f"Best: {best_algo}"),
              use_container_width=True)

    # confusion matrix + ROC for best model
    if extra:
        st.markdown(f"#### {best_algo} — Confusion Matrix & ROC Curve")
        cm_col, roc_col = st.columns(2)

        with cm_col:
            cm = extra[best_algo]["cm"]
            fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.8))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["Predicted\nHealthy","Predicted\nDisease"],
                yticklabels=["Actual\nHealthy","Actual\nDisease"],
                linewidths=0.5, linecolor="white",
                annot_kws={"fontsize": 15, "fontweight": "bold"},
                ax=ax_cm,
            )
            ax_cm.set_title("Confusion Matrix", fontsize=11, pad=8)
            ax_cm.tick_params(labelsize=9)
            fig_cm.tight_layout()
            st.pyplot(fig_cm, use_container_width=True)

        with roc_col:
            fpr = extra[best_algo]["fpr"]
            tpr = extra[best_algo]["tpr"]
            roc_auc = extra[best_algo]["auc"]
            if fpr is not None:
                fig_roc, ax_roc = plt.subplots(figsize=(4.5, 3.8))
                ax_roc.fill_between(fpr, tpr, alpha=0.12, color="#b03a2e")
                ax_roc.plot(fpr, tpr, color="#b03a2e", lw=2.2,
                            label=f"AUC = {roc_auc:.3f}")
                ax_roc.plot([0,1],[0,1], "k--", lw=1, alpha=0.35,
                            label="Random classifier")
                ax_roc.set_xlabel("False Positive Rate", fontsize=9)
                ax_roc.set_ylabel("True Positive Rate", fontsize=9)
                ax_roc.set_title("ROC Curve", fontsize=11, pad=8)
                ax_roc.legend(loc="lower right", fontsize=9)
                ax_roc.spines[["top","right"]].set_visible(False)
                ax_roc.tick_params(labelsize=9)
                fig_roc.tight_layout()
                st.pyplot(fig_roc, use_container_width=True)

    # per-metric chart
    st.markdown("#### All Metrics Across Models")
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    fig_multi, axes = plt.subplots(2, 2, figsize=(10, 6))
    colors_all = ["#b03a2e" if m == best_algo else "#aed6f1" for m in model_names]
    for ax, metric in zip(axes.flatten(), metrics):
        vals = [results[m][metric] for m in model_names]
        ax.barh(model_names, vals, color=colors_all, edgecolor="white", height=0.52)
        for i, (bar_val, bar_h) in enumerate(zip(vals, [0.52]*len(vals))):
            ax.text(bar_val + 0.3, i, f"{bar_val:.1f}", va="center", fontsize=8.5)
        ax.set_xlim(0, 112)
        ax.set_title(metric, fontsize=10, pad=5)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(left=False, labelsize=8.5)
    fig_multi.tight_layout(pad=2)
    st.pyplot(fig_multi, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — DATA INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_insights:
    # load dataset for EDA (use cached if available, else reload)
    if df_cached is not None:
        df_eda = df_cached.copy()
    elif os.path.exists(DATASET):
        df_eda = _load_df()
    else:
        st.warning(f"'{DATASET}' not found — cannot display data insights.")
        st.stop()

    # KPI row
    st.markdown("#### Dataset Overview")
    k1, k2, k3, k4, k5 = st.columns(5)
    kpi_data = [
        ("1,000", "Total Patients"),
        ("12", "Clinical Features"),
        (f"{int(df_eda['target'].sum())}", "Disease Cases"),
        (f"{int((df_eda['target']==0).sum())}", "Healthy Cases"),
        (f"{int(df_eda['gender'].sum())}", "Male Patients"),
    ]
    for col, (val, lbl) in zip([k1,k2,k3,k4,k5], kpi_data):
        col.markdown(f"""
        <div class="kpi">
          <div class="val">{val}</div>
          <div class="lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Age distribution split by target
    st.markdown("#### Age Distribution by Diagnosis")
    fig_age, ax_age = plt.subplots(figsize=(8, 3.5))
    healthy = df_eda[df_eda["target"]==0]["age"]
    disease = df_eda[df_eda["target"]==1]["age"]
    ax_age.hist(healthy, bins=20, alpha=0.6, color="#1e8449", label="Healthy", edgecolor="white")
    ax_age.hist(disease, bins=20, alpha=0.6, color="#b03a2e", label="Disease", edgecolor="white")
    ax_age.set_xlabel("Age (years)", fontsize=9)
    ax_age.set_ylabel("No. of Patients", fontsize=9)
    ax_age.set_title("Age vs Heart Disease — Indian Population", fontsize=10.5, pad=7)
    ax_age.legend(fontsize=9)
    ax_age.spines[["top","right"]].set_visible(False)
    ax_age.tick_params(labelsize=9)
    fig_age.tight_layout()
    st.pyplot(fig_age, use_container_width=True)

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        # Gender vs Target
        st.markdown("#### Gender vs Diagnosis")
        gender_target = df_eda.groupby(["gender","target"]).size().unstack(fill_value=0)
        gender_target.index = ["Female","Male"]
        gender_target.columns = ["Healthy","Disease"]
        fig_g, ax_g = plt.subplots(figsize=(4.5, 3.2))
        gender_target.plot(kind="bar", ax=ax_g, color=["#1e8449","#b03a2e"],
                           edgecolor="white", width=0.55, rot=0)
        ax_g.set_title("Gender vs Heart Disease", fontsize=10.5, pad=7)
        ax_g.set_xlabel(""); ax_g.set_ylabel("Count", fontsize=9)
        ax_g.legend(fontsize=9); ax_g.spines[["top","right"]].set_visible(False)
        ax_g.tick_params(labelsize=9)
        fig_g.tight_layout()
        st.pyplot(fig_g, use_container_width=True)

    with insight_col2:
        # Chest pain type vs Target
        st.markdown("#### Chest Pain Type vs Diagnosis")
        cp_labels = {0:"Asymptomatic", 1:"Atypical\nAngina", 2:"Non-Anginal\nPain", 3:"Typical\nAngina"}
        cp_target = df_eda.groupby(["chestpain","target"]).size().unstack(fill_value=0)
        cp_target.index = [cp_labels[i] for i in cp_target.index]
        cp_target.columns = ["Healthy","Disease"]
        fig_cp, ax_cp = plt.subplots(figsize=(4.5, 3.2))
        cp_target.plot(kind="bar", ax=ax_cp, color=["#1e8449","#b03a2e"],
                       edgecolor="white", width=0.6, rot=15)
        ax_cp.set_title("Chest Pain Type vs Diagnosis", fontsize=10.5, pad=7)
        ax_cp.set_xlabel(""); ax_cp.set_ylabel("Count", fontsize=9)
        ax_cp.legend(fontsize=9); ax_cp.spines[["top","right"]].set_visible(False)
        ax_cp.tick_params(labelsize=9)
        fig_cp.tight_layout()
        st.pyplot(fig_cp, use_container_width=True)

    # Correlation heatmap
    st.markdown("#### Feature Correlation with Target")
    corr = df_eda.corr()[["target"]].drop("target").sort_values("target", ascending=False)
    fig_corr, ax_corr = plt.subplots(figsize=(7, 4))
    bar_colors = ["#b03a2e" if v > 0 else "#2980b9" for v in corr["target"]]
    ax_corr.barh(corr.index, corr["target"], color=bar_colors, edgecolor="white")
    ax_corr.axvline(0, color="#aaa", lw=0.9)
    ax_corr.set_xlabel("Pearson Correlation with Target", fontsize=9)
    ax_corr.set_title("Feature Correlation — Red = positive risk, Blue = protective", fontsize=10.5, pad=7)
    ax_corr.spines[["top","right"]].set_visible(False)
    ax_corr.tick_params(labelsize=9)
    fig_corr.tight_layout()
    st.pyplot(fig_corr, use_container_width=True)

    # Feature importance from Random Forest (retrain on full data)
    st.markdown("#### Feature Importance (Random Forest — Full Dataset)")
    rf_fi = RandomForestClassifier(n_estimators=150, random_state=42)
    X_full = df_eda[feature_cols]
    y_full = df_eda["target"]
    rf_fi.fit(scaler.transform(X_full), y_full)
    importances = (
        pd.Series(rf_fi.feature_importances_, index=feature_cols)
          .sort_values()
    )
    fi_colors = ["#b03a2e" if v >= importances.median() else "#aed6f1"
                 for v in importances.values]
    fig_fi, ax_fi = plt.subplots(figsize=(7, 4.2))
    ax_fi.barh(importances.index, importances.values, color=fi_colors, edgecolor="white")
    ax_fi.set_xlabel("Importance Score", fontsize=9)
    ax_fi.set_title("Feature Importance — Top features in Red", fontsize=10.5, pad=7)
    ax_fi.spines[["top","right"]].set_visible(False)
    ax_fi.tick_params(labelsize=9)
    fig_fi.tight_layout()
    st.pyplot(fig_fi, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("#### About This Project")
    st.markdown("""
This system predicts the likelihood of cardiovascular disease using
clinical parameters from **1,000 Indian patients**. Unlike the original
Cleveland dataset used in most heart disease research, this dataset
reflects the Indian population's typical BP range, cholesterol patterns,
and gender distribution, making predictions more relevant for Indian clinical contexts.
    """)

    st.markdown("---")
    st.markdown("#### Dataset — Feature Reference")
    st.markdown("""
| Feature | Column Name | Description | Indian Context |
|---|---|---|---|
| Age | `age` | Patient age in years | Range: 20–80 |
| Gender | `gender` | 1 = Male, 0 = Female | ~76.5% Male in dataset |
| Chest Pain Type | `chestpain` | 0–3 (0=Asymptomatic, 3=Typical angina) | Asymptomatic most common |
| Resting BP | `restingBP` | Systolic BP in mm Hg | Avg ~152 (higher than Western avg) |
| Serum Cholesterol | `serumcholestrol` | mg/dl | Avg ~311 mg/dl; 0 = not available |
| Fasting Blood Sugar | `fastingbloodsugar` | 1 if FBS > 120 mg/dl | High diabetes prevalence in India |
| Resting ECG | `restingrelectro` | 0=Normal, 1=ST-T abnormality, 2=LVH | — |
| Max Heart Rate | `maxheartrate` | Beats per minute during exercise | — |
| Exercise Angina | `exerciseangia` | 1 = Yes, 0 = No | — |
| ST Depression | `oldpeak` | Induced by exercise vs rest | Range: 0–6.2 in this dataset |
| ST Slope | `slope` | 0–3 (0=Up, 1=Flat, 2=Down, 3=Steep) | — |
| Major Vessels | `noofmajorvessels` | 0–3 coloured by fluoroscopy | 0 = best prognosis |
    """)

    st.markdown("---")
    st.markdown("#### Models Used")
    st.markdown("""
- **Logistic Regression** — fast, interpretable baseline  
- **Random Forest** — ensemble, handles non-linearity well  
- **SVM (RBF kernel)** — strong on small-to-medium datasets  
- **K-Nearest Neighbours** — distance-based, k=7  
- **Decision Tree** — max_depth=6 to prevent overfitting  

Train/test split: **80/20**, stratified by target, `random_state=42`.  
Scaling: **StandardScaler** (fit only on train, transform on test — no data leakage).
    """)

    st.markdown("---")
    st.markdown("""
<div class="disclaimer">
  ⚠️ <b>Disclaimer:</b> This application is developed for academic and educational purposes
  as part of a final-year B.Tech (CSE) project at BACET. It is <b>not intended</b> for clinical
  use or as a substitute for professional medical advice, diagnosis, or treatment.
</div>
""", unsafe_allow_html=True)

    st.markdown("""
---
**Developed by:** Arpan Das · Chandan Kumar Mishra · MD Belal  
**Institution:** BACET — Department of Computer Science & Engineering (CSE)  
**Dataset:** Cardiovascular Disease Dataset — Indian Population (1,000 patients)
    """)
