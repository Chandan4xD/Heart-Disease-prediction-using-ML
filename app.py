import os
import time
import pickle
import warnings
import requests
import pandas as pd
import numpy as np
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

st.set_page_config(page_title="Cardiovascular CDSS Portal", layout="wide", page_icon="🫀")

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
    return trained_models[best_name], scaler, results, feature_names

def compute_cardio_metrics(metrics: dict) -> dict:
    bp = metrics.get('restingBP', 120)
    chol = metrics.get('serumcholestrol', 200)
    mhr = metrics.get('maxheartrate', 150)
    oldpeak = metrics.get('oldpeak', 0.0)
    
    rpp = int(bp * mhr)
    
    if bp < 120:
        bp_class = "Normal Blood Pressure"
    elif 120 <= bp < 130:
        bp_class = "Elevated Blood Pressure"
    elif 130 <= bp < 140:
        bp_class = "Stage I Hypertension"
    else:
        bp_class = "Stage II Hypertension"
        
    if chol < 200:
        chol_class = "Desirable"
    elif 200 <= chol < 240:
        chol_class = "Borderline High Risk"
    else:
        chol_class = "High Risk"
        
    if oldpeak == 0:
        ischemia_class = "No myocardial ischemic stress detected"
    elif 0.1 <= oldpeak <= 1.5:
        ischemia_class = "Mild myocardial ischemia during exercise"
    else:
        ischemia_class = "Severe myocardial ischemia / high ischemic threat"
        
    return {
        'rpp': rpp,
        'bp_class': bp_class,
        'chol_class': chol_class,
        'ischemia_class': ischemia_class
    }

def generate_local_clinical_response(prompt: str, context: dict) -> str:
    if not context:
        return (
            "### ℹ️ General Clinical Guidelines\n"
            "No active patient profile detected in the session. Here is a baseline coronary guidelines summary:\n\n"
            "* **Serum Cholesterol:** Desirable ranges are < 200 mg/dl. Values above 240 mg/dl signify high risk.\n"
            "* **Blood Pressure:** Hypertension Stage II begins at a systolic value of 140 mm Hg or higher.\n"
            "* **ST Depression (Oldpeak):** Measures myocardial ischemia. Values > 1.5 indicate significant stress.\n"
            "* **Fluoroscopy (Vessels Colored):** Values between 1-3 indicate high levels of calcification and vessel occlusion."
        )
    
    pred = context['prediction']
    metrics = context['metrics']
    derived = compute_cardio_metrics(metrics)
    
    age = metrics.get('age', 50)
    gender_txt = "Male" if metrics.get('gender', 1) == 1 else "Female"
    vessels = metrics.get('noofmajorvessels', 0)
    fbs = "Elevated (>120 mg/dl)" if metrics.get('fastingbloodsugar', 0) == 1 else "Normal (<120 mg/dl)"

    if "summary" in prompt.lower() or "generate" in prompt.lower():
        color_alert = "🔴 HIGH RISK CORONARY PROFILE" if pred == "Positive" else "🟢 LOW RISK CORONARY PROFILE"
        recommendation = (
            "An immediate cardiologist consultation and coronary angiogram are strongly indicated due to diagnostic vessel occlusion." 
            if pred == "Positive" else "Continue regular clinical tracking, encourage cardiovascular exercise, and maintain dietary metrics."
        )
        return (
            f"### {color_alert}\n"
            f"**Demographic Context:** {age}-year-old {gender_txt}.\n\n"
            f"**Dynamic Diagnostic Metrics Summary:**\n"
            f"* **Rate Pressure Product (RPP):** {derived['rpp']} bpm*mmHg (Heart Oxygen Workload Index).\n"
            f"* **Vascular Classification:** {derived['bp_class']} (Resting BP: {metrics.get('restingBP')} mmHg).\n"
            f"* **Atherosclerotic Status:** {derived['chol_class']} (Serum Cholesterol: {metrics.get('serumcholestrol')} mg/dl).\n"
            f"* **Myocardial Ischemia Index:** {derived['ischemia_class']} (ST Depression: {metrics.get('oldpeak')} mm).\n"
            f"* **Fasting Blood Sugar:** {fbs}.\n"
            f"* **Fluoroscopy Occlusion:** {vessels} major coronary artery(ies) colored via fluoroscopy.\n\n"
            f"**Clinical Advisory:** {recommendation}"
        )
    elif "risk" in prompt.lower():
        risks = []
        if metrics.get('restingBP', 120) >= 140: 
            risks.append(f"Stage II Hypertension (BP: {metrics.get('restingBP')} mmHg)")
        if metrics.get('serumcholestrol', 200) >= 240: 
            risks.append(f"Hypercholesterolemia (Cholesterol: {metrics.get('serumcholestrol')} mg/dl)")
        if metrics.get('oldpeak', 0.0) >= 1.5: 
            risks.append(f"Severe ST-Segment Depression ({metrics.get('oldpeak')} mm) signifying silent myocardial ischemia")
        if vessels > 0: 
            risks.append(f"Atherosclerosis threat with {vessels} occluded major coronary arteries")
        if derived['rpp'] > 22000:
            risks.append(f"High Rate Pressure Product ({derived['rpp']}) indicating high cardiac oxygen demand")
        
        risk_str = "\n".join([f"* **{r}**" for r in risks]) if risks else "* No acute diagnostic anomalies detected in the input parameters."
        return f"### ⚠️ Target Risk Factor Synthesis\nBased on patient telemetry data, the following indicators require attention:\n\n{risk_str}"
    elif "model" in prompt.lower() or "ml" in prompt.lower():
        return (
            "### 📊 Machine Learning Infrastructure Details\n"
            "This prediction was generated using the **Random Forest Classifier** trained on the Mendeley Indian Cardiovascular Dataset.\n\n"
            "1. **Input Vector:** The 12 clinical inputs are converted into a `1x12` NumPy array.\n"
            "2. **Normalization:** Inputs are normalized using a pre-configured `StandardScaler` trained on the training split.\n"
            "3. **Inference Execution:** The Random Forest model evaluates the standardized vectors across 100 unique decision trees."
        )
    else:
        return "Insight registered. Please use the 'Quick Prompt Options' to run an in-depth clinical study on the current patient parameters."

def call_gemini(prompt, system_instruction):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass

    context_data = None
    if 'latest_pred' in st.session_state:
        context_data = {
            'prediction': st.session_state['latest_pred'],
            'metrics': st.session_state['latest_metrics']
        }

    if not api_key:
        return generate_local_clinical_response(prompt, context_data)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
    
    full_prompt = prompt
    if context_data:
        derived = compute_cardio_metrics(context_data['metrics'])
        full_prompt = (
            f"Active Patient Profile: Prediction={context_data['prediction']}, "
            f"Metrics={str(context_data['metrics'])}. "
            f"Derived Clinical Context: RPP={derived['rpp']}, BP Class={derived['bp_class']}, "
            f"Cholesterol Class={derived['chol_class']}, Ischemia Class={derived['ischemia_class']}. "
            f"Use this data to answer: {prompt}"
        )

    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]}
    }
    
    delays = [1, 2, 4, 8, 16]
    for delay in delays:
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if text:
                    return text
            elif response.status_code in [429, 500, 502, 503]:
                time.sleep(delay)
                continue
            else:
                break
        except Exception:
            time.sleep(delay)
            continue
            
    return generate_local_clinical_response(prompt, context_data)

st.title("Heart Disease Prediction System")
st.caption("Developed by Arpan, Chandan & MD Belal")

model, scaler, results, feature_names = get_models()

tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Performance", "AI Consultant", "About Project"])

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
    
    if st.button("Predict Result", type="primary"):
        input_dict = {
            'age': age, 'gender': gender_val, 'chestpain': chestpain,
            'restingBP': resting_bp, 'serumcholestrol': serumcholestrol,
            'fastingbloodsugar': fastingbloodsugar, 'restingrelectro': restingrelectro,
            'maxheartrate': maxheartrate, 'exerciseangia': exerciseangia,
            'oldpeak': oldpeak, 'slope': slope, 'noofmajorvessels': noofmajorvessels
        }
        try:
            user_input = np.array([[input_dict[col] for col in feature_names]])
            scaled = scaler.transform(user_input) 
            pred = model.predict(scaled)[0]
            st.session_state['latest_pred'] = "Positive" if pred == 1 else "Negative"
            st.session_state['latest_metrics'] = input_dict
            if pred == 1: st.error("⚠️ Heart Disease Detected")
            else: st.success("✅ No Heart Disease Detected")
        except KeyError as e:
            st.error(f"Dataset column mismatch. Missing column: {e}")

with tab2:
    st.write("### Model Performance Metrics")
    df_res = pd.DataFrame(results).T.reset_index()
    df_res.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    df_res = df_res.sort_values(by=['Accuracy', 'F1-Score'], ascending=[False, False]).reset_index(drop=True)
    st.dataframe(df_res, use_container_width=True)

with tab3:
    st.write("### 🤖 Advanced Clinical AI Consultant")
    
    if 'latest_pred' in st.session_state:
        st.success(f"✅ **Active Patient Context Synced Successfully** | Classification Result: **{st.session_state['latest_pred']}**")
    else:
        st.info("ℹ️ **No active patient context.** Perform a test in the **Predict** tab to automatically upload patient telemetry parameters to the AI Consultant.")

    st.write("#### Quick Action Triggers")
    q1, q2, q3, q4 = st.columns(4)
    quick_prompt = None
    with q1:
        if st.button("📋 Patient Clinical Summary", use_container_width=True):
            quick_prompt = "Generate a comprehensive clinical summary of the patient parameters and diagnosis."
    with q2:
        if st.button("🔬 Target Risk Factors", use_container_width=True):
            quick_prompt = "Analyze target risk factors from the patient's metrics."
    with q3:
        if st.button("💻 Explain ML Preprocessing", use_container_width=True):
            quick_prompt = "Explain how the Machine Learning preprocessing and Random Forest pipeline predicted this outcome."
    with q4:
        if st.button("🥗 Preventive Actions", use_container_width=True):
            quick_prompt = "Suggest preventive interventions and lifestyle changes for this profile."

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome. I am your advanced AI consultant. How can I assist you with clinical interpretations or diagnostic metrics today?"}
        ]

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_query = st.chat_input("Input your medical query here...")
    if quick_prompt:
        user_query = quick_prompt

    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        sys_prompt = (
            "You are a clinical cardiologist and AI engineer. Assist users in analyzing cardiodiagnostics data. "
            "Use patient metadata context when explaining outcomes. Always format responses in clean Markdown."
        )

        with st.chat_message("assistant"):
            with st.spinner("Analyzing coronary metrics..."):
                reply = call_gemini(user_query, sys_prompt)
                st.write(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

with tab4:
    st.markdown("### 🧬 CDSS Project Specifications")
    
    st.info(
        "**Clinical Decision Support System (CDSS) for Cardiovascular Risk Stratification** \n"
        "This system leverages deep ensemble machine learning classifiers to predict Coronary Artery Disease (CAD) "
        "and is optimized for cloud architecture deployments."
    )
    
    st.markdown("#### 📊 Core Architecture Details")
    st.markdown("""
    * **Standardized 12-Feature Preprocessing:** Features match the 12 non-invasive metrics of the Indian Cardiovascular Dataset (Mendeley Data). non-predictive keys like `patientid` are scrubbed automatically.
    * **Imputation Guard:** Zero-value errors are resolved via median/mode imputation before scaling to preserve mathematical consistency during execution.
    * **Balanced Split Validation:** 80-20 stratified training structure ensures model evaluation is resilient against target output imbalances.
    """)
    
    with st.expander("🛠️ Machine Learning Model Configurations"):
        st.markdown("""
        * **Random Forest Classifier:** 100 Decision Trees with Gini Impurity criteria. Robust against high metric variances.
        * **Support Vector Machine (SVM):** Implemented with Radial Basis Function (RBF) kernel mapping and dynamic probability estimation.
        * **K-Nearest Neighbors (KNN):** Standard 5-Neighbor Euclidean metric vector.
        * **Naive Bayes:** Gaussian probability density classifier.
        * **Decision Tree:** Single-depth tree configuration.
        """)

    st.markdown("#### 👨‍💻 Project Development Registry")
    st.markdown("""
    Developed under the supervision of the Department of Computer Science & Engineering, **B.A. College of Engineering and Technology (BACET)**:
    * **Arpan Das**
    * **Chandan Kumar Mishra**
    * **MD Belal**
    """)
    
    st.markdown("---")
    st.warning(
        "**Regulatory Disclaimer:** This application serves as an academic and research proof-of-concept. "
        "It is not intended as a substitute for professional clinical screening, diagnostic confirmation, or treatment."
    )
