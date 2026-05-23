import os
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-header {
      text-align: center;
      padding: 1.5rem 0 0.5rem;
  }
  .main-header h1 { font-size: 2rem; font-weight: 700; color: #c0392b; }
  .main-header p  { color: #555; font-size: 0.9rem; }
  .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 500; }
  .predict-card {
      border: 1px solid #e0e0e0;
      border-radius: 12px;
      padding: 1.2rem 1.5rem;
      background: #fafafa;
      margin-bottom: 1rem;
  }
  .result-positive {
      background: #fdecea;
      border-left: 5px solid #c0392b;
      padding: 1rem 1.5rem;
      border-radius: 8px;
      font-size: 1.1rem;
      font-weight: 600;
  }
  .result-negative {
      background: #eafaf1;
      border-left: 5px solid #27ae60;
      padding: 1rem 1.5rem;
      border-radius: 8px;
      font-size: 1.1rem;
      font-weight: 600;
  }
  .metric-card {
      background: #fff;
      border: 1px solid #e8e8e8;
      border-radius: 10px;
      padding: 1rem;
      text-align: center;
  }
  .best-badge {
      background: #eafaf1;
      color: #27ae60;
      border: 1px solid #27ae60;
      border-radius: 20px;
      padding: 2px 12px;
      font-size: 0.78rem;
      font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)


# ── Data & Model Loading ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models...")
def load_all():
    # Try loading pre-saved artifacts first
    if all(os.path.exists(f) for f in ['best_model.pkl', 'scaler.pkl', 'results.pkl']):
        with open('best_model.pkl', 'rb') as f: model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:    scaler = pickle.load(f)
        with open('results.pkl', 'rb') as f:   results = pickle.load(f)
        # For feature importance we still need the df columns
        if os.path.exists('heart.csv'):
            df = pd.read_csv('heart.csv')
            if 'target' not in df.columns and 'output' in df.columns:
                df.rename(columns={'output': 'target'}, inplace=True)
            feature_names = list(df.drop('target', axis=1).columns)
        else:
            feature_names = []
        return model, scaler, results, {}, feature_names

    if not os.path.exists('heart.csv'):
        st.error("❌ 'heart.csv' not found. Place it in the working directory.")
        st.stop()

    df = pd.read_csv('heart.csv')
    if 'target' not in df.columns and 'output' in df.columns:
        df.rename(columns={'output': 'target'}, inplace=True)

    X = df.drop('target', axis=1)
    y = df['target']
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model_defs = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM':                 SVC(kernel='rbf', probability=True, random_state=42),
        'KNN':                 KNeighborsClassifier(n_neighbors=5),
        'Decision Tree':       DecisionTreeClassifier(random_state=42),
    }

    results       = {}
    trained_models = {}
    extra          = {}   # stores confusion matrices, roc data

    for name, clf in model_defs.items():
        clf.fit(X_train_s, y_train)
        preds = clf.predict(X_test_s)
        proba = clf.predict_proba(X_test_s)[:, 1] if hasattr(clf, 'predict_proba') else None

        cm = confusion_matrix(y_test, preds)
        fpr, tpr, _ = roc_curve(y_test, proba) if proba is not None else (None, None, None)
        roc_auc = auc(fpr, tpr) if fpr is not None else None

        results[name] = {
            'Accuracy':  round(accuracy_score(y_test, preds)  * 100, 2),
            'Precision': round(precision_score(y_test, preds) * 100, 2),
            'Recall':    round(recall_score(y_test, preds)    * 100, 2),
            'F1-Score':  round(f1_score(y_test, preds)        * 100, 2),
        }
        trained_models[name] = clf
        extra[name] = {'cm': cm, 'fpr': fpr, 'tpr': tpr, 'auc': roc_auc, 'model': clf}

    best_name  = max(results, key=lambda k: results[k]['Accuracy'])
    best_model = trained_models[best_name]

    # Save artifacts
    with open('best_model.pkl', 'wb') as f: pickle.dump(best_model, f)
    with open('scaler.pkl',     'wb') as f: pickle.dump(scaler, f)
    with open('results.pkl',    'wb') as f: pickle.dump(results, f)

    return best_model, scaler, results, extra, feature_names


model, scaler, results, extra, feature_names = load_all()
best_algo = max(results, key=lambda k: results[k]['Accuracy'])

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🫀 Heart Disease Prediction System</h1>
  <p>Cleveland Clinic Dataset · 5-Model Comparison · Developed at BACET</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Predict", "📊 Performance", "🧬 Insights", "ℹ️ About"])


# ── Tab 1 · Predict ────────────────────────────────────────────────────────────
with tab1:
    st.markdown("#### Enter Clinical Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Patient Info**")
        age    = st.slider("Age", 20, 80, 50)
        sex    = st.radio("Sex", ["Female", "Male"], index=1, horizontal=True)
        sex_v  = 1 if sex == "Male" else 0
        cp     = st.selectbox("Chest Pain Type (cp)",
                              options=[0,1,2,3],
                              format_func=lambda x: {
                                  0:"0 – Typical angina",
                                  1:"1 – Atypical angina",
                                  2:"2 – Non-anginal pain",
                                  3:"3 – Asymptomatic"
                              }[x])
        restecg = st.selectbox("Resting ECG (restecg)",
                               options=[0,1,2],
                               format_func=lambda x: {
                                   0:"0 – Normal",
                                   1:"1 – ST-T abnormality",
                                   2:"2 – LV hypertrophy"
                               }[x])

    with col2:
        st.markdown("**Vitals**")
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        chol     = st.number_input("Cholesterol (mg/dl)",            100, 600, 200)
        fbs      = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                            format_func=lambda x: "Yes" if x else "No", horizontal=True)
        thalach  = st.number_input("Max Heart Rate Achieved", 60, 220, 150)

    with col3:
        st.markdown("**Test Results**")
        exang   = st.radio("Exercise-Induced Angina", [0, 1],
                           format_func=lambda x: "Yes" if x else "No", horizontal=True)
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
        slope   = st.selectbox("ST Slope (slope)",
                               options=[0,1,2],
                               format_func=lambda x: {
                                   0:"0 – Upsloping",
                                   1:"1 – Flat",
                                   2:"2 – Downsloping"
                               }[x])
        ca      = st.selectbox("Major Vessels Coloured (ca)", [0,1,2,3,4])
        thal    = st.selectbox("Thalassemia (thal)",
                               options=[0,1,2,3],
                               format_func=lambda x: {
                                   0:"0 – Normal",
                                   1:"1 – Fixed defect",
                                   2:"2 – Reversible defect",
                                   3:"3 – Unknown"
                               }[x])

    st.markdown("")
    predict_col, _ = st.columns([1, 3])
    with predict_col:
        predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

    if predict_btn:
        user_input = np.array([[age, sex_v, cp, trestbps, chol, fbs,
                                 restecg, thalach, exang, oldpeak, slope, ca, thal]])
        scaled     = scaler.transform(user_input)
        pred       = model.predict(scaled)[0]
        prob       = model.predict_proba(scaled)[0] if hasattr(model, 'predict_proba') else None

        st.markdown("---")
        if pred == 1:
            st.markdown(f"""
            <div class="result-positive">
              ❤️‍🔥 Heart Disease Detected
              {"<br><small>Confidence: " + f"{prob[1]*100:.1f}%" + "</small>" if prob is not None else ""}
            </div>""", unsafe_allow_html=True)
            st.warning("⚠️ This is a screening tool, not a diagnosis. Consult a cardiologist.")
        else:
            st.markdown(f"""
            <div class="result-negative">
              ✅ No Heart Disease Detected
              {"<br><small>Confidence: " + f"{prob[0]*100:.1f}%" + "</small>" if prob is not None else ""}
            </div>""", unsafe_allow_html=True)
            st.info("ℹ️ Result is indicative only. Regular check-ups are recommended.")

        st.caption(f"Model used: **{best_algo}** · Accuracy: {results[best_algo]['Accuracy']}%")


# ── Tab 2 · Performance ────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Model Comparison")

    # Metrics table
    df_res = pd.DataFrame(results).T.reset_index()
    df_res.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    df_res['Best'] = df_res['Model'].apply(lambda x: "⭐ Best" if x == best_algo else "")

    def highlight_best(row):
        return ['background-color: #eafaf1; font-weight: bold'
                if row['Model'] == best_algo else '' for _ in row]

    st.dataframe(
        df_res.style.apply(highlight_best, axis=1)
                    .highlight_max(subset=['Accuracy','Precision','Recall','F1-Score'],
                                   color='#d5f5e3'),
        use_container_width=True, hide_index=True
    )

    # Bar chart comparison
    st.markdown("#### Accuracy by Model")
    fig_bar, ax_bar = plt.subplots(figsize=(9, 3.5))
    models_list = list(results.keys())
    accs = [results[m]['Accuracy'] for m in models_list]
    colors = ['#27ae60' if m == best_algo else '#aed6f1' for m in models_list]
    bars = ax_bar.barh(models_list, accs, color=colors, edgecolor='white', height=0.55)
    for bar, val in zip(bars, accs):
        ax_bar.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=10)
    ax_bar.set_xlim(0, 115)
    ax_bar.set_xlabel('Accuracy (%)', fontsize=10)
    ax_bar.spines[['top','right','left']].set_visible(False)
    ax_bar.tick_params(left=False)
    ax_bar.set_title(f'Best: {best_algo}', fontsize=11, color='#27ae60', pad=8)
    fig_bar.tight_layout()
    st.pyplot(fig_bar, use_container_width=True)

    # Confusion matrix + ROC for best model
    if extra:
        st.markdown(f"#### {best_algo} — Confusion Matrix & ROC Curve")
        cm_col, roc_col = st.columns(2)

        with cm_col:
            cm = extra[best_algo]['cm']
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
            im = ax_cm.imshow(cm, cmap='Blues')
            ax_cm.set_xticks([0,1]); ax_cm.set_yticks([0,1])
            ax_cm.set_xticklabels(['Predicted\nHealthy','Predicted\nDisease'], fontsize=9)
            ax_cm.set_yticklabels(['Actual\nHealthy','Actual\nDisease'], fontsize=9)
            for i in range(2):
                for j in range(2):
                    ax_cm.text(j, i, str(cm[i,j]), ha='center', va='center',
                               fontsize=16, fontweight='bold',
                               color='white' if cm[i,j] > cm.max()/2 else 'black')
            ax_cm.set_title('Confusion Matrix', fontsize=11, pad=8)
            fig_cm.tight_layout()
            st.pyplot(fig_cm, use_container_width=True)

        with roc_col:
            fpr = extra[best_algo]['fpr']
            tpr = extra[best_algo]['tpr']
            roc_auc = extra[best_algo]['auc']
            if fpr is not None:
                fig_roc, ax_roc = plt.subplots(figsize=(4, 3.5))
                ax_roc.plot(fpr, tpr, color='#c0392b', lw=2,
                            label=f'AUC = {roc_auc:.3f}')
                ax_roc.plot([0,1],[0,1],'k--', lw=1, alpha=0.4)
                ax_roc.set_xlabel('False Positive Rate', fontsize=9)
                ax_roc.set_ylabel('True Positive Rate', fontsize=9)
                ax_roc.set_title('ROC Curve', fontsize=11, pad=8)
                ax_roc.legend(loc='lower right', fontsize=9)
                ax_roc.spines[['top','right']].set_visible(False)
                fig_roc.tight_layout()
                st.pyplot(fig_roc, use_container_width=True)


# ── Tab 3 · Insights ──────────────────────────────────────────────────────────
with tab3:
    if not os.path.exists('heart.csv'):
        st.warning("heart.csv not found — cannot render insights.")
    else:
        df_data = pd.read_csv('heart.csv')
        if 'target' not in df_data.columns and 'output' in df_data.columns:
            df_data.rename(columns={'output': 'target'}, inplace=True)

        st.markdown("#### Dataset Overview")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records",   len(df_data))
        m2.metric("Features",         df_data.shape[1] - 1)
        m3.metric("Disease Cases",    int(df_data['target'].sum()))
        m4.metric("Healthy Cases",    int((df_data['target'] == 0).sum()))

        st.markdown("#### Feature Importance (Random Forest)")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        X_all = df_data.drop('target', axis=1)
        y_all = df_data['target']
        X_s = scaler.transform(X_all)
        rf.fit(X_s, y_all)
        importances = pd.Series(rf.feature_importances_, index=X_all.columns).sort_values()

        fig_fi, ax_fi = plt.subplots(figsize=(8, 5))
        colors_fi = ['#c0392b' if v > importances.median() else '#aed6f1'
                     for v in importances.values]
        importances.plot(kind='barh', ax=ax_fi, color=colors_fi, edgecolor='white')
        ax_fi.set_xlabel('Importance Score', fontsize=10)
        ax_fi.set_title('Feature Importance — Random Forest', fontsize=11, pad=8)
        ax_fi.spines[['top','right']].set_visible(False)
        fig_fi.tight_layout()
        st.pyplot(fig_fi, use_container_width=True)

        st.markdown("#### Correlation with Target")
        corr = df_data.corr()[['target']].drop('target').sort_values('target', ascending=False)
        fig_corr, ax_corr = plt.subplots(figsize=(5, 5))
        colors_c = ['#c0392b' if v > 0 else '#2980b9' for v in corr['target']]
        ax_corr.barh(corr.index, corr['target'], color=colors_c, edgecolor='white')
        ax_corr.axvline(0, color='black', lw=0.8, alpha=0.4)
        ax_corr.set_xlabel('Pearson Correlation', fontsize=10)
        ax_corr.set_title('Feature Correlation with Target', fontsize=11, pad=8)
        ax_corr.spines[['top','right']].set_visible(False)
        fig_corr.tight_layout()
        st.pyplot(fig_corr, use_container_width=True)


# ── Tab 4 · About ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown("""
    #### About This Project

    This system uses the **Cleveland Clinic Heart Disease Dataset** to predict the presence
    of heart disease based on 13 clinical features.

    **Models evaluated:** Logistic Regression, Random Forest, SVM, KNN, Decision Tree

    **Best model** is automatically selected by accuracy on a held-out 20% test split
    (stratified, `random_state=42`).

    ---

    **Feature Reference**

    | Feature | Description |
    |---|---|
    | age | Age in years |
    | sex | 1 = Male, 0 = Female |
    | cp | Chest pain type (0–3) |
    | trestbps | Resting blood pressure |
    | chol | Serum cholesterol (mg/dl) |
    | fbs | Fasting blood sugar > 120 mg/dl |
    | restecg | Resting ECG results (0–2) |
    | thalach | Max heart rate achieved |
    | exang | Exercise-induced angina |
    | oldpeak | ST depression induced by exercise |
    | slope | Slope of peak exercise ST segment |
    | ca | Number of major vessels (0–4) |
    | thal | Thalassemia (0–3) |

    ---

    **Disclaimer:** This tool is for educational and screening purposes only.
    It is not a substitute for professional medical diagnosis.

    ---
    Developed by **Arpan Das, Chandan Kumar Mishra & MD Belal** · BACET
    """)
