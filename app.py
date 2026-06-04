import os
import time
import warnings
import requests
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Cardiovascular CDSS", 
    layout="wide", 
    page_icon="🫀",
    initial_sidebar_state="expanded"
)

@st.cache_resource(show_spinner=False)
def load_and_train():
    if not os.path.exists('heart.csv'):
        st.error("Error: 'heart.csv' file not found.")
        st.stop()

    df = pd.read_csv('heart.csv')
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')
    
    if 'patientid' in df.columns:
        df.drop('patientid', axis=1, inplace=True)
        
    rename_dict = {
        'sex': 'gender', 'cp': 'chestpain', 'trestbps': 'restingBP', 
        'chol': 'serumcholestrol', 'fbs': 'fastingbloodsugar', 
        'restecg': 'restingrelectro', 'thalach': 'maxheartrate', 
        'exang': 'exerciseangia', 'ca': 'noofmajorvessels', 'Classification': 'target'
    }
    df.rename(columns=rename_dict, inplace=True)
    
    X = df.drop('target', axis=1)
    y = df['target']
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    accuracy = f"{accuracy_score(y_test, preds) * 100:.2f}%"
    
    perf = {
        'Random Forest': {'Accuracy': accuracy}
    }
    importance = dict(zip(feature_names, model.feature_importances_))

    return model, scaler, feature_names, accuracy, perf, importance

def get_medical_indices(m):
    if not m:
        return {}
    sbp = m.get('restingBP', 120)
    hr = m.get('maxheartrate', 150)
    chol = m.get('serumcholestrol', 200)
    oldpeak = m.get('oldpeak', 0.0)
    vessels = m.get('noofmajorvessels', 0)
    age = m.get('age', 50)
    
    rpp = int(sbp * hr)
    max_hr_expected = 220 - age
    chronotropic_index = round((hr / max_hr_expected) * 100, 1) if max_hr_expected > 0 else 100.0
    
    if sbp < 120: bp_stage = "Normal Baseline"
    elif sbp < 130: bp_stage = "Elevated State"
    elif sbp < 140: bp_stage = "Stage 1 Hypertension"
    else: bp_stage = "Stage 2 Hypertension"
        
    if chol < 200: lipid_stage = "Normal Profile"
    elif chol < 240: lipid_stage = "Borderline High"
    else: lipid_stage = "High Risk Level"
        
    if oldpeak == 0: ischemia_stage = "No ST Segment Depression"
    elif oldpeak <= 1.5: ischemia_stage = "Mild to Moderate Ischemia"
    else: ischemia_stage = "Severe Ischemia Risk"
        
    return {
        "rpp": rpp,
        "chronotropic_index": f"{chronotropic_index}%",
        "bp_stage": bp_stage,
        "lipid_stage": lipid_stage,
        "ischemia_stage": ischemia_stage,
        "vessels": f"{vessels} blocked major vessels"
    }

def run_groq(chat_history, system_prompt):
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
        if not api_key:
            return "Error: GROQ_API_KEY identifier missing from Streamlit Secrets vault."

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "system", "content": system_prompt}] + chat_history,
            "temperature": 0.2,
            "max_tokens": 1500
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

model, scaler, feature_names, accuracy, performance_matrix, feature_importances = load_and_train()

st.title("Cardiovascular Clinical Decision Support System")
st.caption("Developed by Arpan Das, Chandan Kumar Mishra & MD Belal")

with st.sidebar:
    st.header("Patient Metrics")
    st.markdown("---")
    
    age = st.slider("Age", 18, 100, 52)
    gender_text = st.radio("Sex", ["Female (0)", "Male (1)"], index=1, horizontal=True)
    gender = 1 if "Male" in gender_text else 0
    
    chestpain = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                            format_func=lambda x: {0: "0: Asymptomatic", 1: "1: Typical Angina", 2: "2: Atypical Angina", 3: "3: Non-Anginal"}[x])
    
    restingBP = st.number_input("Resting Blood Pressure (mmHg)", 80, 220, 125)
    serumcholestrol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 210)
    fastingbloodsugar = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1], horizontal=True)
    
    restingrelectro = st.selectbox("Resting ECG Configuration", [0, 1, 2],
                             format_func=lambda x: {0: "0: Normal", 1: "1: ST-T Anomaly", 2: "2: Hypertrophy"}[x])
    
    maxheartrate = st.number_input("Max Heart Rate Achieved (bpm)", 60, 220, 145)
    exerciseangia = st.radio("Exercise Induced Angina", [0, 1], horizontal=True)
    
    oldpeak = st.slider("ST Depression Depth", 0.0, 7.0, 1.0, step=0.1)
    slope = st.selectbox("ST Slope Configuration", [1, 2, 3],
                               format_func=lambda x: {1: "1: Upsloping", 2: "2: Flat Line", 3: "3: Downsloping"}[x])
    
    noofmajorvessels = st.selectbox("Blocked Major Vessels", [0, 1, 2, 3])

    st.markdown("---")
    run_diagnostic = st.button("Run Diagnostic Prediction", type="primary", use_container_width=True)

if run_diagnostic or 'patient_state' in st.session_state:
    
    if run_diagnostic:
        metrics_map = {
            'age': age, 'gender': gender, 'chestpain': chestpain, 'restingBP': restingBP,
            'serumcholestrol': serumcholestrol, 'fastingbloodsugar': fastingbloodsugar, 'restingrelectro': restingrelectro,
            'maxheartrate': maxheartrate, 'exerciseangia': exerciseangia, 'oldpeak': oldpeak,
            'slope': slope, 'noofmajorvessels': noofmajorvessels
        }
        
        row_vector = []
        for col in feature_names:
            row_vector.append(metrics_map.get(col, 0))
            
        scaled_vector = scaler.transform(np.array([row_vector]))
        prediction_output = model.predict(scaled_vector)[0]
        probability_output = model.predict_proba(scaled_vector)[0][1]
            
        st.session_state['patient_state'] = {
            'metrics': metrics_map,
            'prediction': "POSITIVE (Risk Identified)" if prediction_output == 1 else "NEGATIVE (Normal Thresholds)",
            'probability': probability_output
        }

    tab_dashboard, tab_chat, tab_metrics = st.tabs([
        "📊 Live Diagnostic Dashboard", 
        "🧠 AI Cardiology Consultant", 
        "🔬 Model Evaluation Metrics"
    ])

    state = st.session_state['patient_state']
    indices = get_medical_indices(state['metrics'])

    with tab_dashboard:
        st.markdown("### Model Diagnostics Result")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                label="Risk Probability", 
                value=f"{state['probability'] * 100:.2f}%",
                delta="Elevated Warning Risk" if state['probability'] > 0.5 else "Safe Core Level"
            )
        with c2:
            st.metric(label="Rate Pressure Product (Workload)", value=f"{indices['rpp']} mmHg·bpm")
        with c3:
            st.metric(label="Selected Algorithm Node", value="Random Forest")

        if state['probability'] > 0.5:
            st.error(f"🚨 **High Risk Alert:** System classification output is {state['prediction']}.")
        else:
            st.success(f"✅ **Normal Status Clearance:** System classification output is {state['prediction']}.")

        st.markdown("#### Clinical Context Benchmarks")
        b1, b2, b3 = st.columns(3)
        b1.info(f"**BP Stage:** \n\n {indices['bp_stage']}")
        b2.info(f"**Lipid Level:** \n\n {indices['lipid_stage']}")
        b3.info(f"**Ischemia Status:** \n\n {indices['ischemia_stage']}")

    with tab_chat:
        st.markdown("### Interactive Reasoning Engine")
        
        m_row1, m_row2, m_row3 = st.columns(3)
        override_input = None
        
        if m_row1.button("📋 Patient Case Differential Summary", use_container_width=True):
            override_input = "Generate a full differential diagnostic case summary evaluating the metrics context map."
        if m_row2.button("🔬 Pathophysiological Stress Analysis", use_container_width=True):
            override_input = "Analyze the patient ischemic markers, chronotropic capabilities, and rate pressure profiles."
        if m_row3.button("💻 Mathematical Prediction Audit", use_container_width=True):
            override_input = "Audit the data science configuration and explain how feature split bounds determined this risk score."

        if "chat_history_v6" not in st.session_state:
            st.session_state.chat_history_v6 = [
                {"role": "assistant", "content": "Consultant system online and fully synchronized with the active telemetry array. Ask any clinical or structural pipeline questions below."}
            ]

        for chat_msg in st.session_state.chat_history_v6:
            with st.chat_message(chat_msg["role"]):
                st.markdown(chat_msg["content"])

        user_input = st.chat_input("Ask a clinical query or inspect model criteria configurations...")
        if override_input:
            user_input = override_input

        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.chat_history_v6.append({"role": "user", "content": user_input})

            system_prompt = (
                "ROLE AND CONTEXT IDENTITY PROTOCOL:\n"
                "You are an Elite Interventional Cardiologist and Distinguished Data Scientist running inside a Clinical Decision Support System.\n\n"
                "INTELLIGENT RESPONSE MANDATE:\n"
                "You must carefully evaluate the user's specific text input. Do not force every response into a frozen template if the user is asking a conversational question, greeting you, or clarifying an abstract concept. Instead, adapt your reasoning style dynamically to directly answer what the user typed.\n\n"
                "EXECUTION SCHEMAS:\n"
                "- If the user says 'hello', greets you, or asks general state questions (e.g. 'how are you'): Respond professionally as an active, stateful AI consultation core, acknowledge your synchronization state with the patient telemetry, and ask how you can help analyze the patient matrix.\n"
                "- If the user clicks a macro button or asks for a formal report: Structure your response completely using the publication-grade medical analysis headings below (### 1. 🔬 ADVANCED HEMODYNAMIC WORKLOAD KINETICS, ### 2. 🫀 ISCHEMIC WAVEFORM CONFIGURATION MATRIX, ### 3. 💻 DATA-SCIENCE INFRASTRUCTURE & ENSEMBLE PIPELINE AUDIT).\n\n"
                "=== SYNCHRONIZED PATIENT TELEMETRY CHART ===\n"
                f"- Model Output Verdict: {state['prediction']}\n"
                f"- Model Risk Probability: {state['probability'] * 100:.2f}%\n"
                f"- Selected Model Node: Random Forest Classifier\n"
                f"- Rate Pressure Product (RPP): {indices['rpp']} mmHg·bpm\n"
                f"- Chronotropic Capacity Performance Index: {indices['chronotropic_index']}\n"
                f"- Blood Pressure Tier: {indices['bp_stage']}\n"
                f"- Cholesterol Risk Tier: {indices['lipid_stage']}\n"
                f"- Ischemia Waveform Profile: {indices['ischemia_stage']}\n"
                f"- Calcified Vessels: {indices['vessels']}\n"
                f"- Raw Input Data payload: {str(state['metrics'])}\n\n"
                "Maintain an authoritative, brilliant medical and data science engineering tone. Never break character."
            )

            active_window = st.session_state.chat_history_v6[-6:]

            with st.chat_message("assistant"):
                with st.spinner("Processing token array streams over Groq LPU cluster..."):
                    response_payload = run_groq(active_window, system_prompt)
                    st.markdown(response_payload)
                    
            st.session_state.chat_history_v6.append({"role": "assistant", "content": response_payload})

    with tab_metrics:
        st.markdown("### Model Benchmark Performance")
        st.dataframe(pd.DataFrame(performance_matrix).T, use_container_width=True)
        
        st.markdown("#### Global Feature Importance Weights Map")
        st.json(feature_importances)

else:
    st.info("💡 **Notice:** Machine learning arrays and Groq link connections are completely initialized. Configure the parameter inputs in the sidebar and click 'Run Diagnostic Prediction' to generate analytical tracking maps.")
