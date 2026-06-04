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
    page_title="PulseCheck", 
    layout="wide", 
    page_icon="🫀",
    initial_sidebar_state="expanded"
)

@st.cache_resource(show_spinner=False)
def train_model():
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
    feats = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    acc = f"{accuracy_score(y_test, preds) * 100:.2f}%"
    
    perf = {'Random Forest': {'Accuracy': acc}}
    imp = dict(zip(feats, model.feature_importances_))

    return model, scaler, feats, acc, perf, imp

def calc_metrics(m):
    if not m:
        return {}
    sbp = m.get('restingBP', 120)
    hr = m.get('maxheartrate', 150)
    chol = m.get('serumcholestrol', 200)
    op = m.get('oldpeak', 0.0)
    vess = m.get('noofmajorvessels', 0)
    age = m.get('age', 50)
    
    rpp = int(sbp * hr)
    max_hr = 220 - age
    chrono_idx = round((hr / max_hr) * 100, 1) if max_hr > 0 else 100.0
    
    if sbp < 120: bp_stg = "Normal"
    elif sbp < 130: bp_stg = "Elevated"
    elif sbp < 140: bp_stg = "Stage 1 High BP"
    else: bp_stg = "Stage 2 High BP"
        
    if chol < 200: lipid_stg = "Normal"
    elif chol < 240: lipid_stg = "Borderline High"
    else: lipid_stg = "High Risk"
        
    if op == 0: isch_stg = "Normal Blood Flow"
    elif op <= 1.5: isch_stg = "Mildly Reduced Blood Flow"
    else: isch_stg = "Severely Blocked Blood Flow"
        
    return {
        "rpp": rpp,
        "chrono_idx": f"{chrono_idx}%",
        "bp_stage": bp_stg,
        "lipid_stage": lipid_stg,
        "ischemia_stage": isch_stg,
        "vessels": f"{vess} clear vessels" if vess == 0 else f"{vess} blocked vessels"
    }

def run_groq(chat_history, sys_prompt):
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
        if not api_key:
            return "Error: GROQ_API_KEY missing from secrets."

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "system", "content": sys_prompt}] + chat_history,
            "temperature": 0.3,
            "max_tokens": 1200
        }
        
        res = requests.post(url, headers=headers, json=payload, timeout=15)
        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
        else:
            return f"Error {res.status_code}: {res.text}"
    except Exception as e:
        return f"Error: {str(e)}"

model, scaler, feats, acc, perf, imp = train_model()

st.title("PulseCheck")
st.caption("AI Smart Assistant for Personal Heart Health Insights")

with st.sidebar:
    st.header("Your Vitals")
    st.markdown("---")
    
    age = st.slider("Age", 18, 100, 52)
    gender_txt = st.radio("Sex", ["Female", "Male"], index=1, horizontal=True)
    gender = 1 if "Male" in gender_txt else 0
    
    cp = st.selectbox("Chest Pain Experience", [0, 1, 2, 3], 
                      format_func=lambda x: {0: "No Pain", 1: "Severe Pain", 2: "Mild Pain", 3: "Uncomfortable Pressure"}[x])
    
    sbp = st.number_input("Blood Pressure (Systolic)", 80, 220, 125)
    chol = st.number_input("Cholesterol Level", 100, 600, 210)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], index=0, horizontal=True)
    fbs_val = 1 if fbs == "Yes" else 0
    
    ecg = st.selectbox("Resting ECG Result", [0, 1, 2],
                       format_func=lambda x: {0: "Normal Baseline", 1: "Slight Wave Change", 2: "Enlarged Heart Muscle"}[x])
    
    hr = st.number_input("Highest Exercise Heart Rate Achieved", 60, 220, 145)
    exang = st.radio("Chest Pain Triggered by Exercise", ["No", "Yes"], index=0, horizontal=True)
    exang_val = 1 if exang == "Yes" else 0
    
    op = st.slider("ECG Stress Shift Depth (ST Change)", 0.0, 7.0, 1.0, step=0.1)
    slope = st.selectbox("ECG Wave Slope Shape", [1, 2, 3],
                         format_func=lambda x: {1: "Sloping Up (Better)", 2: "Flat (Warning)", 3: "Sloping Down (Risk)"}[x])
    
    vess = st.selectbox("Number of Blocked Major Blood Vessels", [0, 1, 2, 3])

    st.markdown("---")
    run_diag = st.button("Check My Heart Health Status", type="primary", use_container_width=True)

if run_diag or 'state' in st.session_state:
    
    if run_diag:
        data = {
            'age': age, 'gender': gender, 'chestpain': cp, 'restingBP': sbp,
            'serumcholestrol': chol, 'fastingbloodsugar': fbs_val, 'restingrelectro': ecg,
            'maxheartrate': hr, 'exerciseangia': exang_val, 'oldpeak': op,
            'slope': slope, 'noofmajorvessels': vess
        }
        
        vector = [data.get(col, 0) for col in feats]
        vector_scaled = scaler.transform(np.array([vector]))
        pred = model.predict(vector_scaled)[0]
        prob = model.predict_proba(vector_scaled)[0][1]
            
        st.session_state['state'] = {
            'metrics': data,
            'prediction': "High Heart Disease Risk Warning" if pred == 1 else "Normal Low Risk Status",
            'probability': prob
        }

    tab_dash, tab_chat, tab_perf = st.tabs([
        "📊 Health Summary Dashboard", 
        "💬 Ask Your AI Companion", 
        "⚙️ Core System Blueprint Accuracy"
    ])

    state = st.session_state['state']
    idx = calc_metrics(state['metrics'])

    with tab_dash:
        st.markdown("### Heart Health Evaluation Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Calculated Heart Risk Likelihood", 
                value=f"{state['probability'] * 100:.1f}%",
                delta="Elevated Warning Risk" if state['probability'] > 0.5 else "Safe Level"
            )
        with col2:
            st.metric(label="Myocardial Heart Workload Score", value=f"{idx['rpp']}")
        with col3:
            st.metric(label="Overall Testing Model Accuracy", value=acc)

        if state['probability'] > 0.5:
            st.error(f"⚠️ **Attention Required:** Our AI assistant flagged an elevated condition matching potential heart strain patterns: {state['prediction']}.")
        else:
            st.success(f"✅ **Looking Good:** Your current inputs sit comfortably inside a low-risk profile: {state['prediction']}.")

        st.markdown("#### Simple Health Explanations")
        b1, b2, b3 = st.columns(3)
        b1.info(f"**Blood Pressure Zone:** \n\n {idx['bp_stage']}")
        b2.info(f"**Cholesterol Health Status:** \n\n {idx['lipid_stage']}")
        b3.info(f"**Heart Muscle Blood Flow:** \n\n {idx['ischemia_stage']}")

    with tab_chat:
        st.markdown("### Chat with PulseCheck AI")
        
        m1, m2, m3 = st.columns(3)
        override = None
        
        if m1.button("📋 Summarize My Case in Simple Words", use_container_width=True):
            override = "Please explain my patient numbers, my risk percentage, and my general heart profile in simple English without medical jargon."
        if m2.button("🏃‍♂️ Explain My Heart Strain & Exercise Stats", use_container_width=True):
            override = "Explain how my heart handles exercise based on my peak heart rate, workload scores, and the ECG slope lines."
        if m3.button("💻 How did the computer find this risk score?", use_container_width=True):
            override = "Explain in easy terms how the machine learning model weighed my inputs to compute this specific score."

        if "history" not in st.session_state:
            st.session_state.history = [
                {"role": "assistant", "content": "Hello! I am PulseCheck, your personal heart health AI companion. Your vitals are connected directly to my memory. Ask me anything about your metrics!"}
            ]

        for msg in st.session_state.history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_in = st.chat_input("Ask a quick question about your heart stats...")
        if override:
            user_in = override

        if user_in:
            with st.chat_message("user"):
                st.markdown(user_in)
            st.session_state.history.append({"role": "user", "content": user_in})

            sys_prompt = (
                "ROLE AND SYSTEM INSTRUCTIONS:\n"
                "You are PulseCheck, an incredibly supportive, smart, friendly, and practical personal health coach and expert data assistant. Your job is to talk to everyday people about their heart health statistics.\n\n"
                "COMMUNICATION RULES:\n"
                "1. STRICTLY ELIMINATE jargon. Never use technical phrases like 'neuro-symbolic', 'hyperplane', 'Gini impurity', 'subendocardial ischemia', 'pathophysiological metrics', or 'rate pressure product kinetics' in your chat conversation text.\n"
                "2. Speak like a empathetic human friend. Instead of 'myocardial workload surrogate', say 'how hard your heart pumps blood when you push it'. Instead of 'ischemic segment depression', say 'signs of temporary oxygen drop to the heart walls'.\n"
                "3. If the user says 'hello' or asks a casual question, greet them warmly, confirm you see their numbers, and offer clear advice.\n"
                "4. If they press a macro button or ask for details, break it down clearly into 3 basic sections: '1. 🫀 Your Heart Workload & Fitness', '2. 🩸 Blood Flow & Oxygen Status', and '3. 🖥️ Computer Model Explanation'. Keep every section easy to read and intuitive.\n\n"
                "=== CONNECTED HEALTH CHART ===\n"
                f"- Computer Model Risk Conclusion: {state['prediction']}\n"
                f"- Risk Probability Score: {state['probability'] * 100:.1f}%\n"
                f"- Pumping Workload Score: {idx['rpp']}\n"
                f"- Exercise Capacity limit reached: {idx['chrono_idx']}\n"
                f"- Blood Pressure Zone: {idx['bp_stage']}\n"
                f"- Cholesterol Assessment: {idx['lipid_stage']}\n"
                f"- Exercise Oxygen Flow Check: {idx['ischemia_stage']}\n"
                f"- Target Blocked Arteries found: {idx['vessels']}\n"
                f"- Raw Vitals List: {str(state['metrics'])}\n"
            )

            window = st.session_state.history[-6:]

            with st.chat_message("assistant"):
                with st.spinner("Talking to PulseCheck core..."):
                    payload_res = run_groq(window, sys_prompt)
                    st.markdown(payload_res)
                    
            st.session_state.history.append({"role": "assistant", "content": payload_res})

    with tab_perf:
        st.markdown("### System Dashboard Performance Details")
        st.dataframe(pd.DataFrame(perf).T, use_container_width=True)
        
        st.markdown("#### System Input Parameter Value Priorities")
        st.json(imp)

else:
    st.info("💡 **Welcome:** PulseCheck is ready. Adjust your vitals in the left panel and click 'Check My Heart Health Status' to visualize your metrics.")
