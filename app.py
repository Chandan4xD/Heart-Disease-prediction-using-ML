import os
import pickle
import warnings
import requests
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

@st.cache_resource(show_spinner=False)
def get_models():
    if all(os.path.exists(f) for f in ['best_model.pkl', 'scaler.pkl', 'results.pkl', 'feature_names.pkl']):
        with open('best_model.pkl', 'rb') as f: model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
        with open('results.pkl', 'rb') as f: results = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f: feature_names = pickle.load(f)
        return model, scaler, results, feature_names

    if not os.path.exists('heart.csv'):
        st.error("⚠️ Error: 'heart.csv' not found.")
        st.stop()

    df = pd.read_csv('heart.csv')
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')
    
    if 'patientid' in df.columns:
        df.drop('patientid', axis=1, inplace=True)
    if 'Classification' in df.columns and 'target' not in df.columns:
        df.rename(columns={'Classification': 'target'}, inplace=True)
    
    X, y = df.drop('target', axis=1), df['target']
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB()
    }

    results = {}
    trained_models = {}
    for name, clf in models.items():
        clf.fit(X_train_s, y_train)
        preds = clf.predict(X_test_s)
            
        results[name] = {
            'Accuracy': round(accuracy_score(y_test, preds) * 100, 2),
            'Precision': round(precision_score(y_test, preds) * 100, 2),
            'Recall': round(recall_score(y_test, preds) * 100, 2),
            'F1-Score': round(f1_score(y_test, preds) * 100, 2),
        }
        trained_models[name] = clf

    best_name = max(results, key=lambda k: results[k]['Accuracy'])
    
    with open('best_model.pkl', 'wb') as f: pickle.dump(trained_models[best_name], f)
    with open('scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
    with open('results.pkl', 'wb') as f: pickle.dump(results, f)
    with open('feature_names.pkl', 'wb') as f: pickle.dump(feature_names, f)
    
    return trained_models[best_name], scaler, results, feature_names

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
            return "Error: GROQ_API_KEY identifier missing from secrets."

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

st.title("Heart Disease Prediction System")
st.caption("Developed by Arpan, Chandan & MD Belal")

model, scaler, results, feature_names = get_models()
best_algo = max(results, key=lambda k: results[k]['Accuracy'])

tab1, tab2, tab3 = st.tabs(["Predict", "Performance", "About"])

with tab1:
    st.write("### Input Clinical Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 20, 80, 50)
        gender = st.radio("Gender", ["Female (0)", "Male (1)"], index=1, horizontal=True)
        gender_val = 1 if "Male" in gender else 0
        chestpain = st.selectbox("Chest Pain Type (chestpain)", [0, 1, 2, 3])
        restingrelectro = st.selectbox("Resting ECG (restingrelectro)", [0, 1, 2])
    with c2:
        resting_bp = st.number_input("Resting BP (resting BP)", 90, 200, 120)
        serumcholestrol = st.number_input("Serum Cholesterol (serumcholestrol)", 120, 600, 200)
        fastingbloodsugar = st.radio("Fasting Blood Sugar > 120 (fastingbloodsugar)", [0, 1], horizontal=True)
        slope = st.selectbox("ST Slope (slope)", [1, 2, 3])
    with c3:
        maxheartrate = st.number_input("Max Heart Rate (maxheartrate)", 70, 210, 150)
        exerciseangia = st.radio("Exercise Angina (exerciseangia)", [0, 1], horizontal=True)
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.2, 1.0)
        noofmajorvessels = st.selectbox("Major Vessels (noofmajorvessels)", [0, 1, 2, 3])
    
    predict_clicked = st.button("Predict Result", type="primary")
    
    if predict_clicked:
        input_dict = {
            'age': age,
            'gender': gender_val,
            'chestpain': chestpain,
            'restingBP': resting_bp,
            'serumcholestrol': serumcholestrol,
            'fastingbloodsugar': fastingbloodsugar,
            'restingrelectro': restingrelectro,
            'maxheartrate': maxheartrate,
            'exerciseangia': exerciseangia,
            'oldpeak': oldpeak,
            'slope': slope,
            'noofmajorvessels': noofmajorvessels
        }
        
        try:
            user_input = np.array([[input_dict[col] for col in feature_names]])
            scaled = scaler.transform(user_input) 
            pred = model.predict(scaled)[0]
            prob = model.predict_proba(scaled)[0][1] if hasattr(model, "predict_proba") else (1.0 if pred == 1 else 0.0)
                
            st.session_state['state'] = {
                'metrics': input_dict,
                'prediction': "POSITIVE (Risk Identified)" if pred == 1 else "NEGATIVE (Normal Thresholds)",
                'probability': prob
            }
        except KeyError as e:
            st.error(f"Dataset column mismatch. Could not find column: {e}. Please ensure you are using the correct Mendeley dataset.")

    if 'state' in st.session_state:
        state = st.session_state['state']
        idx = calc_metrics(state['metrics'])
        
        st.markdown("---")
        st.write("### Diagnostic Status")
        
        if "POSITIVE" in state['prediction']:
            st.error(f"⚠️ Heart Disease Detected (Risk Probability: {state['probability'] * 100:.2f}%)")
        else:
            st.success(f"✅ No Heart Disease Detected (Risk Probability: {state['probability'] * 100:.2f}%)")
            
        m1, m2, m3 = st.columns(3)
        m1.info(f"**Blood Pressure:** {idx['bp_stage']}")
        m2.info(f"**Cholesterol:** {idx['lipid_stage']}")
        m3.info(f"**Heart Muscle Flow:** {idx['ischemia_stage']}")
        
        st.markdown("---")
        st.write("💬 **PulseCheck AI Assistant**")
        
        mac1, mac2 = st.columns(2)
        override = None
        if mac1.button("📋 Summarize Case Profile", use_container_width=True):
            override = "Please explain my patient numbers, my risk percentage, and my general heart profile in simple English without medical jargon."
        if mac2.button("🏃‍♂️ Explain Heart Workload & Stress", use_container_width=True):
            override = "Explain how my heart handles exercise based on my peak heart rate, workload scores, and the ECG slope lines."

        if "history" not in st.session_state:
            st.session_state.history = [
                {"role": "assistant", "content": "Hello! I am PulseCheck, your personal heart health AI companion. Your vitals are connected directly to my memory. Ask me anything about your metrics!"}
            ]

        for msg in st.session_state.history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_in = st.chat_input("Ask an AI question about your diagnostic values...")
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
                f"- Active Pipeline Engine: {best_algo}\n"
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
                with st.spinner("Analyzing stats..."):
                    payload_res = run_groq(window, sys_prompt)
                    st.markdown(payload_res)
                    
            st.session_state.history.append({"role": "assistant", "content": payload_res})
            st.rerun()

with tab2:
    st.write("### Model Performance Metrics")
    df_res = pd.DataFrame(results).T.reset_index()
    df_res.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    df_res = df_res.sort_values(by=['Accuracy', 'F1-Score'], ascending=[False, False]).reset_index(drop=True)
    st.dataframe(df_res, use_container_width=True)

with tab3:
    st.markdown("### 🧬 Project Overview")
    st.info(
        "This application is a **Clinical Decision Support System (CDSS)** powered by Machine Learning. "
        "It is designed to evaluate the likelihood of cardiovascular disease based on standard non-invasive clinical metrics."
    )
    
    st.markdown("### 📊 Dataset & Architecture")
    st.markdown("""
    * **Data Source:** Indian Cardiovascular Disease Dataset (Mendeley Data).
    * **Feature Engineering:** 12 curated clinical attributes localized for accurate demographic prediction.
    * **Inference Engine:** An automated pipeline evaluating multiple classification algorithms (Random Forest, SVM, Decision Tree, KNN, Naive Bayes) to deploy the optimal predictive model.
    """)
    
    st.markdown("### 👨‍💻 Development Team")
    st.markdown("""
    Engineered at **B.A. College of Engineering and Technology (BACET)** by:
    * **Arpan Das**
    * **Chandan Kumar Mishra**
    * **MD Belal**
    """)
    
    st.markdown("---")
    st.warning(
        "**Clinical Disclaimer:** This software is developed for academic and research purposes. "
        "It is not a substitute for professional medical advice, diagnosis, or treatment."
    )
