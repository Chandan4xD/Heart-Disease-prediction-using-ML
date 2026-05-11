# =============================================================
# Heart Disease Prediction - Streamlit Web App
# BA College of Engineering and Technology
# Run: streamlit run app.py
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# ----------------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #c0392b, #922b21);
        padding: 25px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p  { margin: 5px 0 0; opacity: 0.85; font-size: 0.95rem; }

    .metric-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07);
    }
    .metric-card h3 { margin: 0; font-size: 1.8rem; color: #c0392b; }
    .metric-card p  { margin: 4px 0 0; color: #555; font-size: 0.85rem; }

    .result-positive {
        background: #fdecea;
        border-left: 5px solid #e74c3c;
        padding: 20px;
        border-radius: 8px;
        margin-top: 15px;
    }
    .result-negative {
        background: #eafaf1;
        border-left: 5px solid #2ecc71;
        padding: 20px;
        border-radius: 8px;
        margin-top: 15px;
    }
    .result-positive h2, .result-negative h2 { margin: 0 0 8px; }
    .result-positive p,  .result-negative p  { margin: 0; color: #444; }

    .stButton > button {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #c0392b;
        border-bottom: 2px solid #fdecea;
        padding-bottom: 6px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# LOAD / TRAIN MODEL (cached)
# ----------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_or_train():
    """Load pretrained model, or train fresh from UCI dataset."""

    if os.path.exists('best_model.pkl') and os.path.exists('scaler.pkl'):
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('results.pkl', 'rb') as f:
            results = pickle.load(f)
        return model, scaler, results

    # --- train fresh ---
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
                'thalach','exang','oldpeak','slope','ca','thal','target']
    df = pd.read_csv(url, names=columns, na_values='?')
    df.dropna(inplace=True)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    df['ca']   = df['ca'].astype(int)
    df['thal'] = df['thal'].astype(int)

    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        'Decision Tree':       DecisionTreeClassifier(random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM':                 SVC(kernel='rbf', probability=True, random_state=42),
        'KNN':                 KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    }

    results = {}
    trained = {}
    for name, m in models.items():
        m.fit(X_train_s, y_train)
        yp = m.predict(X_test_s)
        results[name] = {
            'Accuracy':  round(accuracy_score(y_test, yp)  * 100, 2),
            'Precision': round(precision_score(y_test, yp) * 100, 2),
            'Recall':    round(recall_score(y_test, yp)    * 100, 2),
            'F1-Score':  round(f1_score(y_test, yp)        * 100, 2),
        }
        trained[name] = m

    best_name  = max(results, key=lambda k: results[k]['Accuracy'])
    best_model = trained[best_name]

    with open('best_model.pkl', 'wb') as f: pickle.dump(best_model, f)
    with open('scaler.pkl',     'wb') as f: pickle.dump(scaler,     f)
    with open('results.pkl',    'wb') as f: pickle.dump(results,    f)

    return best_model, scaler, results


# ----------------------------------------------------------
# HEADER
# ----------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>❤️ Heart Disease Prediction System</h1>
    <p>BA College of Engineering and Technology &nbsp;|&nbsp;
       Department of CSE &nbsp;|&nbsp; 8th Semester &nbsp;|&nbsp;
       Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------
with st.spinner("Loading ML models... please wait ⏳"):
    best_model, scaler, results = load_or_train()

best_name = max(results, key=lambda k: results[k]['Accuracy'])

# ----------------------------------------------------------
# TABS
# ----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Model Performance", "ℹ️ About"])

# ============================================================
# TAB 1 – PREDICTION
# ============================================================
with tab1:
    st.markdown("### Enter Patient Details")
    st.caption("Fill in the clinical parameters below and click **Predict** to get the result.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-title">👤 Demographics</div>', unsafe_allow_html=True)
        age  = st.slider("Age", 20, 80, 50)
        sex  = st.radio("Sex", ["Male", "Female"], horizontal=True)
        sex_val = 1 if sex == "Male" else 0

    with col2:
        st.markdown('<div class="section-title">🩺 Clinical Readings</div>', unsafe_allow_html=True)
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        chol     = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
        thalach  = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
        oldpeak  = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)

    with col3:
        st.markdown('<div class="section-title">🔬 Test Results</div>', unsafe_allow_html=True)
        cp = st.selectbox("Chest Pain Type", [
            "0 – Typical Angina",
            "1 – Atypical Angina",
            "2 – Non-anginal Pain",
            "3 – Asymptomatic"
        ])
        cp_val = int(cp[0])

        fbs     = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No (0)", "Yes (1)"], horizontal=True)
        fbs_val = int(fbs[-2])

        restecg = st.selectbox("Resting ECG", [
            "0 – Normal",
            "1 – ST-T Wave Abnormality",
            "2 – Left Ventricular Hypertrophy"
        ])
        restecg_val = int(restecg[0])

        exang = st.radio("Exercise Induced Angina", ["No (0)", "Yes (1)"], horizontal=True)
        exang_val = int(exang[-2])

        slope = st.selectbox("Slope of Peak ST Segment", [
            "1 – Upsloping",
            "2 – Flat",
            "3 – Downsloping"
        ])
        slope_val = int(slope[0])

        ca = st.selectbox("Major Vessels Colored (0–3)", [0, 1, 2, 3])

        thal = st.selectbox("Thalassemia", [
            "3 – Normal",
            "6 – Fixed Defect",
            "7 – Reversible Defect"
        ])
        thal_val = int(thal[0])

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Heart Disease")

    if predict_btn:
        input_data = np.array([[
            age, sex_val, cp_val, trestbps, chol,
            fbs_val, restecg_val, thalach, exang_val,
            oldpeak, slope_val, ca, thal_val
        ]])
        input_scaled = scaler.transform(input_data)
        prediction   = best_model.predict(input_scaled)[0]
        probability  = best_model.predict_proba(input_scaled)[0][1] * 100 if hasattr(best_model, 'predict_proba') else None

        if prediction == 1:
            prob_str = f"Confidence: **{probability:.1f}%**" if probability else ""
            st.markdown(f"""
            <div class="result-positive">
                <h2>⚠️ Heart Disease DETECTED</h2>
                <p>{prob_str} &nbsp;|&nbsp; Model: <strong>{best_name}</strong></p>
                <p style="margin-top:10px;">Please consult a cardiologist immediately for further evaluation and diagnosis.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            prob_str = f"Confidence: **{100 - probability:.1f}%**" if probability else ""
            st.markdown(f"""
            <div class="result-negative">
                <h2>✅ No Heart Disease Detected</h2>
                <p>{prob_str} &nbsp;|&nbsp; Model: <strong>{best_name}</strong></p>
                <p style="margin-top:10px;">Results look healthy! Maintain a balanced diet and regular exercise.</p>
            </div>
            """, unsafe_allow_html=True)

        st.info("⚠️ **Disclaimer:** This tool is for educational purposes only and does not replace professional medical advice.")

# ============================================================
# TAB 2 – MODEL PERFORMANCE
# ============================================================
with tab2:
    st.markdown("### Model Comparison Results")

    results_df = pd.DataFrame(results).T.reset_index()
    results_df.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']

    # Metric cards
    best = results_df.loc[results_df['Accuracy'].idxmax()]
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><h3>{best["Accuracy"]}%</h3><p>Accuracy</p></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><h3>{best["Precision"]}%</h3><p>Precision</p></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><h3>{best["Recall"]}%</h3><p>Recall</p></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><h3>{best["F1-Score"]}%</h3><p>F1-Score</p></div>', unsafe_allow_html=True)

    st.markdown(f"<p style='text-align:center; color:#888; margin-top:8px;'>Best model: <strong>{best['Model']}</strong></p>", unsafe_allow_html=True)
    st.markdown("---")

    # Table
    st.dataframe(
        results_df.style.highlight_max(subset=['Accuracy','Precision','Recall','F1-Score'],
                                        color='#fdecea'),
        use_container_width=True, hide_index=True
    )

    # Bar chart
    st.markdown("### Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ['#e74c3c' if m == best['Model'] else '#aab7b8' for m in results_df['Model']]
    bars = ax.bar(results_df['Model'], results_df['Accuracy'], color=colors, edgecolor='white', width=0.5)
    ax.set_ylim(70, 100)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Accuracy Comparison', fontweight='bold')
    for bar, val in zip(bars, results_df['Accuracy']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.tick_params(axis='x', labelsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ============================================================
# TAB 3 – ABOUT
# ============================================================
with tab3:
    st.markdown("""
    ### About This Project

    **Heart Disease Prediction Using Machine Learning** is a final-year B.Tech project developed at
    **BA College of Engineering and Technology**, Department of Computer Science & Engineering (8th Semester).

    ---

    #### 📂 Dataset
    - **Source:** Cleveland Heart Disease Dataset — UCI Machine Learning Repository
    - **Records:** ~303 patients (after preprocessing)
    - **Features:** 13 clinical attributes
    - **Target:** Heart Disease Present (1) / Absent (0)

    #### 🤖 Models Used
    | Model | Description |
    |---|---|
    | Decision Tree | Rule-based classification tree |
    | Random Forest | Ensemble of 100 decision trees |
    | SVM | Hyperplane-based classification |
    | KNN | k=5 nearest neighbours |
    | Logistic Regression | Binary classification via sigmoid |

    #### 🏗️ Tech Stack
    - **Python** · **Scikit-learn** · **Pandas** · **NumPy** · **Matplotlib** · **Streamlit**

    #### 👥 Team
    | Name | Reg. No | Role |
    |---|---|---|
    | Arpan Das | 22010440004 | Dataset Analysis & EDA |
    | Chandan Kumar Mishra | 22010440005 | Model Building & Evaluation |
    | MD Belal | 22010440010 | Streamlit App & Documentation |
    """)

# ----------------------------------------------------------
# FOOTER
# ----------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.8rem;'>"
    "❤️ Heart Disease Prediction | BA College of Engineering and Technology | 8th Semester CSE"
    "</p>",
    unsafe_allow_html=True
)
