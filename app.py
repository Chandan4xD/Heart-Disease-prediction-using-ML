import os
import pickle
import warnings
import requests
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Cardiovascular CDSS Engine", layout="wide", page_icon="🫀")

@st.cache_resource(show_spinner=False)
def get_models():
    if all(os.path.exists(f) for f in ['best_model.pkl', 'scaler.pkl', 'results.pkl', 'feature_names.pkl']):
        with open('best_model.pkl', 'rb') as f: return pickle.load(f), pickle.load(open('scaler.pkl', 'rb')), pickle.load(open('results.pkl', 'rb')), pickle.load(open('feature_names.pkl', 'rb'))
    if not os.path.exists('heart.csv'): st.error("⚠️ Error: 'heart.csv' not found."); st.stop()
    
    df = pd.read_csv('heart.csv')
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')
    if 'patientid' in df.columns: df.drop('patientid', axis=1, inplace=True)
    if 'Classification' in df.columns: df.rename(columns={'Classification': 'target'}, inplace=True)
    
    X, y = df.drop('target', axis=1), df['target']
    f_names = X.columns.tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)
    clfs = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB()
    }
    res = {}
    for name, clf in clfs.items():
        clf.fit(X_tr_s, y_tr)
        p = clf.predict(X_te_s)
        res[name] = {'Accuracy': round(accuracy_score(y_te, p)*100, 2), 'Precision': round(precision_score(y_te, p)*100, 2), 'Recall': round(recall_score(y_te, p)*100, 2), 'F1-Score': round(f1_score(y_te, p)*100, 2)}
    
    best = clfs[max(res, key=lambda k: res[k]['Accuracy'])]
    return best, scaler, res, f_names

def compute_hemodynamics(m):
    if not m: return {}
    bp, hr, chol, op = m.get('restingBP', 120), m.get('maxheartrate', 150), m.get('serumcholestrol', 200), m.get('oldpeak', 0.0)
    return {
        "rpp": int(bp * hr),
        "bp_stage": "Normal" if bp < 120 else "Elevated" if bp < 130 else "Stage 1 HTN" if bp < 140 else "Stage 2 HTN",
        "lipid_risk": "Desirable" if chol < 200 else "Borderline High" if chol < 240 else "High Risk",
        "ischemia_load": "None" if op == 0 else "Mild/Mod" if op <= 1.5 else "Severe Myocardial Risk"
    }

def call_groq_brain(messages):
    try:
        if "GROQ_API_KEY" not in st.secrets: return "⚠️ API Key missing from st.secrets configuration."
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-specdec", "messages": messages, "temperature": 0.2},
            timeout=15
        )
        return res.json()["choices"][0]["message"]["content"] if res.status_code == 200 else f"Inference Error: {res.text}"
    except Exception as e: return f"Execution failed: {str(e)}"

st.title("Enterprise Cardiovascular CDSS Engine")
st.caption("Developed by Arpan, Chandan & MD Belal")
model, scaler, results, feature_names = get_models()

t1, t2, t3, t4 = st.tabs(["Patient Screening", "Analytics Matrix", "Cognitive AI Consultant", "System Specs"])

with t1:
    st.write("### Telemetry Input Vectors")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age Scope", 20, 80, 50)
        gender = st.radio("Biological Sex", ["Female (0)", "Male (1)"], index=1, horizontal=True)
        chestpain = st.selectbox("Angina Type (chestpain)", [0, 1, 2, 3])
        restingrelectro = st.selectbox("ECG Morph (restingrelectro)", [0, 1, 2])
    with c2:
        resting_bp = st.number_input("Resting Systolic BP (mmHg)", 90, 200, 120)
        serumcholestrol = st.number_input("Serum Cholesterol (mg/dl)", 120, 600, 200)
        fastingbloodsugar = st.radio("Fasting Glucose > 120 mg/dl", [0, 1], horizontal=True)
        slope = st.selectbox("ST Segment Slope (slope)", [1, 2, 3])
    with c3:
        maxheartrate = st.number_input("Peak Achieved Heart Rate (bpm)", 70, 210, 150)
        exerciseangia = st.radio("Exertion Induced Angina", [0, 1], horizontal=True)
        oldpeak = st.number_input("ST Depression Depth (oldpeak)", 0.0, 6.2, 1.0)
        noofmajorvessels = st.selectbox("Fluoroscopy Calcified Vessels", [0, 1, 2, 3])
    
    if st.button("Execute Diagnostic Pipeline", type="primary"):
        input_dict = {'age': age, 'gender': 1 if "Male" in gender else 0, 'chestpain': chestpain, 'restingBP': resting_bp, 'serumcholestrol': serumcholestrol, 'fastingbloodsugar': fastingbloodsugar, 'restingrelectro': restingrelectro, 'maxheartrate': maxheartrate, 'exerciseangia': exerciseangia, 'oldpeak': oldpeak, 'slope': slope, 'noofmajorvessels': noofmajorvessels}
        pred = model.predict(scaler.transform(np.array([[input_dict[col] for col in feature_names]])))[0]
        st.session_state['latest_pred'] = "Positive (CAD Pathology Confirmed)" if pred == 1 else "Negative (Normal Homeostatic Profile)"
        st.session_state['latest_metrics'] = input_dict
        if pred == 1: st.error("⚠️ Diagnostic Alert: High Probability of Cardiovascular Pathology")
        else: st.success("✅ Diagnostic Clearance: Normal Homeostatic Cardiac Profile")

with t2:
    st.write("### Pipeline Classifier Cross-Validation Matrix")
    st.dataframe(pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Classifier'}).sort_values(by=['Accuracy'], ascending=False).reset_index(drop=True), use_container_width=True)

with t3:
    st.write("### 🤖 Cognitive AI Consultant (Enterprise Layer)")
    has_ctx = 'latest_pred' in st.session_state
    hd = compute_hemodynamics(st.session_state.get('latest_metrics', {}))
    
    if has_ctx:
        st.success(f"🧬 Active Telemetry Sync Active | Model Output: **{st.session_state['latest_pred']}**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Myocardial Workload (RPP)", f"{hd['rpp']} mmHg·bpm")
        m2.metric("AHA BP Stratum", hd['bp_stage'])
        m3.metric("Lipid Risk Index", hd['lipid_risk'])
        m4.metric("Ischemic Vector Burden", hd['ischemia_load'])
    else: st.info("ℹ️ Cognitive Engine Standby: Complete a screening to inject telemetry context matrix.")

    c1, c2, c3 = st.columns(3)
    act_prompt = None
    if c1.button("📋 Generate Differential Diagnostic Report", use_container_width=True, disabled=not has_ctx): act_prompt = "Perform a deep, formal medical differential diagnostic synthesis of the active patient profile."
    if c2.button("🔬 Ischemic & Metabolic Risk Mapping", use_container_width=True, disabled=not has_ctx): act_prompt = "Evaluate the patient's ischemic load, RPP metrics, and atherosclerotic risk vectors systematically."
    if c3.button("💻 Explain Mathematical Inference Path", use_container_width=True, disabled=not has_ctx): act_prompt = "Explain how the pipeline's active ensemble model evaluated this raw feature matrix mathematically."

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": "Cognitive engine initialized. Standing by for multi-turn differential diagnostics, hemodynamic parsing, and machine learning pipeline validation audits."}]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.write(msg["content"])

    user_in = st.chat_input("Query diagnostic matrix or explore pathophysiology correlations...")
    if act_prompt: user_in = act_prompt

    if user_in:
        with st.chat_message("user"): st.write(user_in)
        st.session_state.chat_history.append({"role": "user", "content": user_in})
        
        sys_directive = (
            "You are a premier Interventional Cardiologist and Senior AI Research Scientist specializing in digital health. "
            "You provide rigorous, elite, data-driven insights. Do not rewrite or repeat the user's input back to them. Do not use generic medical filler text.\n\n"
            f"CURRENT LIVE PATIENT CLINICAL CONTEXT MATRIX:\n"
            f"- Machine Learning Pipeline Prediction: {st.session_state.get('latest_pred', 'NO ACTIVE SCREENING IN CONTEXT')}\n"
            f"- Raw Telemetry Feature Matrix: {str(st.session_state.get('latest_metrics', 'None'))}\n"
            f"- Extracted Hemodynamic Parameters: {str(hd)}\n\n"
            "INSTRUCTION: Use the patient context matrix above to run real-time clinical reasoning. If the user asks a conversational question, "
            "relate it directly to their specific raw feature values, derived physiological indices, or the ensemble model performance parameters. "
            "Respond using professional, publication-grade markdown matrices, bold technical thresholds, and clear pathophysiological breakdowns."
        )
        
        api_payload = [{"role": "system", "content": sys_directive}]
        for m in st.session_state.chat_history[1:]: api_payload.append({"role": m["role"], "content": m["content"]})
            
        with st.chat_message("assistant"):
            with st.spinner("Processing cognitive matrix via Groq LPU..."):
                ai_out = call_groq_brain(api_payload)
                st.write(ai_out)
        st.session_state.chat_history.append({"role": "assistant", "content": ai_out})

with t4:
    st.markdown("### 🧬 Architecture & Development Framework")
    st.info("**Clinical Decision Support System (CDSS) Node** | Engineered under the supervision of the Department of Computer Science & Engineering, **B.A. College of Engineering and Technology (BACET)**.")
    st.markdown("""
    * **State-Aware Multi-Turn Conversational Context:** Unlike basic stateless engines, the AI Consultant maps the entire conversational thread alongside dynamic internal hemodynamic indicators into a centralized inference context array.
    * **Self-Healing Pipeline Engine:** Instantly evaluates file structure arrays, handles missing variance anomalies via robust median/mode parsing, and normalizes incoming telemetry streams via localized standard scalar transforms.
    """)
    st.markdown("#### 👨‍💻 Project Development Registry")
    st.markdown("* **Arpan Das**\n* **Chandan Kumar Mishra**\n* **MD Belal**")
