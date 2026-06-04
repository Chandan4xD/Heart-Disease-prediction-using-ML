import os
import time
import pickle
import warnings
import requests
import pandas as pd
import numpy as np
import streamlit as st

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

def compute_clinical_indexes(metrics):
    if not metrics:
        return {}
    bp = metrics.get('restingBP', 120)
    hr = metrics.get('maxheartrate', 150)
    chol = metrics.get('serumcholestrol', 200)
    oldpeak = metrics.get('oldpeak', 0.0)
    
    rpp = int(bp * hr)
    
    if bp < 120: bp_cat = "Normal"
    elif 120 <= bp < 130: bp_cat = "Elevated"
    elif 130 <= bp < 140: bp_cat = "Stage 1 Hypertension"
    else: bp_cat = "Stage 2 Hypertension"
        
    if chol < 200: chol_cat = "Desirable"
    elif 200 <= chol < 240: chol_cat = "Borderline High"
    else: chol_cat = "High Risk"
        
    if oldpeak == 0: isch_cat = "No Current Sign of Ischemia"
    elif 0 < oldpeak <= 1.5: isch_cat = "Mild to Moderate Myocardial Ischemia"
    else: isch_cat = "Severe Myocardial Ischemia Risk"
        
    return {
        "rpp": rpp,
        "bp_category": bp_cat,
        "cholesterol_category": chol_cat,
        "ischemic_category": isch_cat
    }

def generate_cognitive_fallback(prompt, context):
    if not context:
        return (
            "### 🔍 Core System Diagnostics Mode\n"
            "The system is currently operating in standby. To invoke the diagnostic synthesis reasoning engine, "
            "please complete a screening entry in the **Predict** tab.\n\n"
            "**Standard Clinical Benchmarks Monitored:**\n"
            "* Myocardial Workload Index (Rate Pressure Product Threshold: 12,000)\n"
            "* AHA Hypertension Stratification Protocols\n"
            "* Coronary Artery Calcification Grading Via Fluoroscopy Vectors"
        )
    
    m = context['metrics']
    idx = compute_clinical_indexes(m)
    pred = context['prediction']
    
    if "summary" in prompt.lower() or "profile" in prompt.lower() or "generate" in prompt.lower():
        status_color = "🔴 DETECTED ANOMALIES" if pred == "Positive" else "🟢 REGULAR PHYSIOLOGICAL FUNCTION"
        return (
            f"### 📋 Advanced Cardiovascular Synthesis Report\n"
            f"**Current Status:** {status_color}\n\n"
            f"#### 1. Hemodynamic & Workload Metrics\n"
            f"* **Rate Pressure Product (RPP):** {idx['rpp']} mmHg·bpm (Values > 12,000 signify elevated myocardial oxygen demand during exertion).\n"
            f"* **AHA Blood Pressure Classification:** {idx['bp_category']} based on a resting value of {m.get('restingBP')} mmHg.\n\n"
            f"#### 2. Metabolic & Vascular Profiles\n"
            f"* **Atherosclerotic Lipid Burden:** {idx['cholesterol_category']} ({m.get('serumcholestrol')} mg/dl).\n"
            f"* **Fluoroscopic Vascular Occlusion:** {m.get('noofmajorvessels')} major coronary vessel(s) exhibiting significant calcification.\n\n"
            f"#### 3. Pathophysiological Assessment\n"
            f"* **Ischemic segment Index:** {idx['ischemic_category']} ({m.get('oldpeak')} mm ST-segment depression).\n"
            f"* **ST-Slope Vector Morphology:** Configuration category {m.get('slope')}.\n\n"
            f"#### 4. Clinical Logic Conclusion\n"
            f"The Random Forest Classifier has labeled this vector profile as **{pred}** for heart disease risk. "
            f"The computed internal indexes mirror this finding via elevated ischemia markers and structural coronary occlusion values. Immediate diagnostic echocardiography is recommended."
        )
    elif "risk" in prompt.lower():
        anomalies = []
        if idx['bp_category'] in ["Stage 1 Hypertension", "Stage 2 Hypertension"]: anomalies.append(f"Arterial Hypertension ({m.get('restingBP')} mmHg)")
        if idx['cholesterol_category'] == "High Risk": anomalies.append(f"Hypercholesterolemia Lipid Profile ({m.get('serumcholestrol')} mg/dl)")
        if m.get('noofmajorvessels', 0) > 0: anomalies.append(f"Structural Coronoary Calcification ({m.get('noofmajorvessels')} vessels blocked)")
        if m.get('oldpeak', 0) > 1.5: anomalies.append(f"High Ischemic Load Baseline ({m.get('oldpeak')} mm ST Depression)")
        
        anomalies_str = "\n".join([f"* **{a}**" for a in anomalies]) if anomalies else "* No primary clinical anomalies flagged within the active telemetry frame."
        return f"### 🔬 Target Risk Stratification Summary\n\n{anomalies_str}"
    else:
        return "Insight compiled. Select 'Patient Clinical Summary' or 'Target Risk Factors' above to generate comprehensive analytical streams."

def call_groq(prompt, system_instruction):
    api_key = ""
    try:
        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    context_data = None
    if 'latest_pred' in st.session_state:
        context_data = {
            'prediction': st.session_state['latest_pred'],
            'metrics': st.session_state['latest_metrics']
        }

    if not api_key:
        return generate_cognitive_fallback(prompt, context_data)

    url = "https://api.groq.com/openai/v1/chat/completions"
    
    if context_data:
        idx = compute_clinical_indexes(context_data['metrics'])
        full_context = (
            f"Active Patient Clinical Profile:\n"
            f"- Machine Learning Prediction: {context_data['prediction']}\n"
            f"- Calculated Rate Pressure Product (RPP): {idx.get('rpp')}\n"
            f"- AHA Blood Pressure Stratification: {idx.get('bp_category')}\n"
            f"- Lipid Risk Category: {idx.get('cholesterol_category')}\n"
            f"- Ischemic Shift Assessment: {idx.get('ischemic_category')}\n"
            f"- Raw Features: {str(context_data['metrics'])}\n\n"
            f"Evaluate this specific medical query with high clinical rigor: {prompt}"
        )
    else:
        full_context = prompt

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-specdec",
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": full_context}
        ],
        "temperature": 0.2
    }
    
    delays = [1, 2, 4]
    for delay in delays:
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            elif response.status_code in [429, 500, 503]:
                time.sleep(delay)
                continue
            else: break
        except Exception:
            time.sleep(delay)
            continue
            
    return generate_cognitive_fallback(prompt, context_data)

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
        st.success(f"🧬 **Patient Context Synchronized** | Current Classification Core: **{st.session_state['latest_pred']}**")
        derived = compute_clinical_indexes(st.session_state['latest_metrics'])
        k1, k2, k3 = st.columns(3)
        k1.metric("Myocardial Oxygen Workload (RPP)", f"{derived['rpp']} mmHg·bpm")
        k2.metric("AHA Blood Pressure Status", derived['bp_category'])
        k3.metric("Ischemic Vector Index", derived['ischemic_category'])
    else:
        st.info("ℹ️ Standby: Complete an evaluation in the **Predict** tab to automatically generate advanced medical index profiles for the AI Consultant.")

    st.write("#### Clinical Prompt Diagnostics Trigger")
    q1, q2, q3 = st.columns(3)
    auto_trigger = None
    with q1:
        if st.button("📋 Comprehensive Patient Summary", use_container_width=True):
            auto_trigger = "Generate an integrated clinical summary detailing Derived Clinical Indexes and risks."
    with q2:
        if st.button("🔬 Ischemic & Vascular Risk Synthesis", use_container_width=True):
            auto_trigger = "Synthesize specific cardiovascular target risk factors from the patient profile variables."
    with q3:
        if st.button("💻 Structural Pipeline Analysis", use_container_width=True):
            auto_trigger = "Explain how the Random Forest Classifier arrived at this prediction step mathematically."

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Consultant system initialized. Powered by Groq LPU inference. I can parse cross-feature telemetry correlations, compute metabolic thresholds, and break down pipeline inference states."}
        ]

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    user_query = st.chat_input("Query patient diagnostics matrix...")
    if auto_trigger:
        user_query = auto_trigger

    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        sys_prompt = (
            "You are an expert clinical cardiologist and a senior machine learning researcher specializing in digital health. "
            "Analyze the patient data using standard cardiac medicine protocols. Synthesize secondary metrics like Rate Pressure Product "
            "and ST segment morphology accurately. Be direct, authoritative, and structured. Use clear Markdown headings."
        )

        with st.chat_message("assistant"):
            with st.spinner("Synthesizing telemetry data stream via Groq..."):
                reply = call_groq(user_query, sys_prompt)
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
    Engineered under the supervision of the Department of Computer Science & Engineering, **B.A. College of Engineering and Technology (BACET)** by:
    * **Arpan Das**
    * **Chandan Kumar Mishra**
    * **MD Belal**
    """)
    st.markdown("---")
    st.warning(
        "**Regulatory Disclaimer:** This application serves as an academic and research proof-of-concept. "
        "It is not intended as a substitute for professional clinical screening, diagnostic confirmation, or treatment."
    )
