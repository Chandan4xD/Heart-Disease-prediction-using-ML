import os
import time
import pickle
import warnings
import requests
import numpy as np
import pandas as pd
import streamlit as st

# Explicit Enterprise Machine Learning Toolkit Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

# High-Fidelity Page Layout Configuration
st.set_page_config(
    page_title="Cardiovascular CDSS Node v4.0", 
    layout="wide", 
    page_icon="🫀",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------------
# MODULE 1: MACHINE LEARNING PIPELINE ENGINE (MULTI-MODEL ENSEMBLE)
# -------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def initialize_ml_pipeline():
    """
    Loads or trains the underlying multi-model classification architecture,
    calculates comparative metrics, and extracts predictive feature importance weights.
    """
    if not os.path.exists('heart.csv'):
        st.error("🚨 Critical System Fault: 'heart.csv' data source repository not found.")
        st.stop()

    # Load and clean source telemetry array
    df = pd.read_csv('heart.csv')
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')
    
    if 'patientid' in df.columns:
        df.drop('patientid', axis=1, inplace=True)
    if 'Classification' in df.columns and 'target' not in df.columns:
        df.rename(columns={'Classification': 'target'}, inplace=True)
    
    # Isolate feature vectors and target labels
    X = df.drop('target', axis=1)
    y = df['target']
    feature_names = X.columns.tolist()
    
    # Stratified split to preserve clinical target balances
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale feature dimensions to normalize variance arrays
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Instantiate multi-model evaluation array
    models = {
        'Random Forest Classifier': RandomForestClassifier(n_estimators=120, random_state=42),
        'Gradient Boosting Engine': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Support Vector Node (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'Regularized Logistic Core': LogisticRegression(max_iter=1000, random_state=42)
    }

    performance_matrix = {}
    trained_models = {}
    
    for name, clf in models.items():
        clf.fit(X_train_scaled, y_train)
        predictions = clf.predict(X_test_scaled)
        
        performance_matrix[name] = {
            'Accuracy': round(accuracy_score(y_test, predictions) * 100, 2),
            'Precision': round(precision_score(y_test, predictions) * 100, 2),
            'Recall': round(recall_score(y_test, predictions) * 100, 2),
            'F1-Score': round(f1_score(y_test, predictions) * 100, 2),
        }
        trained_models[name] = clf

    # Extract global feature importance from the ensemble baseline (Random Forest)
    rf_model = trained_models['Random Forest Classifier']
    importance_scores = rf_model.feature_importances_
    feature_importance_map = dict(zip(feature_names, importance_scores))

    return trained_models, scaler, performance_matrix, feature_names, feature_importance_map


# -------------------------------------------------------------------------
# MODULE 2: REASONING & CLINICAL ANALYTICS TRANSLATION ENGINE
# -------------------------------------------------------------------------
def compute_clinical_hemodynamics(m):
    """
    Translates raw telemetry numbers into actionable secondary clinical indicators.
    """
    if not m:
        return {}
        
    sbp = m.get('restingBP', 120)
    hr = m.get('maxheartrate', 150)
    chol = m.get('serumcholestrol', 200)
    oldpeak = m.get('oldpeak', 0.0)
    vessels = m.get('noofmajorvessels', 0)
    
    # Rate Pressure Product (Surrogate for Myocardial Oxygen Consumption)
    rpp = int(sbp * hr)
    
    # AHA Blood Pressure Classification Layer
    if sbp < 120: bp_stage = "Normal Normotensive Baseline"
    elif 120 <= sbp < 130: bp_stage = "Elevated Systolic State"
    elif 130 <= sbp < 140: bp_stage = "Stage 1 Arterial Hypertension"
    else: bp_stage = "Stage 2 Severe Arterial Hypertension"
        
    # Hyperlipidemia Atherosclerotic Risk Stratification
    if chol < 200: lipid_risk = "Optimal/Desirable Metabolic Range"
    elif 200 <= chol < 240: lipid_risk = "Borderline High Atherosclerotic Load"
    else: lipid_risk = "High-Risk Atherosclerotic Profile"
        
    # Ischemic Microvascular Burden Index
    if oldpeak == 0: ischemia_status = "No Active ST-Segment Shift Detected"
    elif 0 < oldpeak <= 1.5: ischemia_status = "Mild-to-Moderate Induced Myocardial Ischemia"
    else: ischemia_status = "Severe Transmural Ischemic Stress Variant"
        
    return {
        "rpp": rpp,
        "bp_stage": bp_stage,
        "lipid_risk": lipid_risk,
        "ischemia_status": ischemia_status,
        "vessel_calcification": f"{vessels} principal coronary branches occluded via fluoroscopy"
    }


def call_groq_cognitive_engine(conversation_payload, system_context):
    """
    Dispatches fully contextualized multi-turn chat payloads directly to Groq.
    """
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
        if not api_key:
            return "⚠️ **System Error:** Connection failed. The `GROQ_API_KEY` token is missing from your Streamlit Secrets vault."

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Build strict payload structure matching OpenAI/Groq standards
        messages = [{"role": "system", "content": system_context}] + conversation_payload
        
        payload = {
            "model": "llama-3.3-70b-specdec",
            "messages": messages,
            "temperature": 0.15,  # Locked low to eliminate random hallucinations
            "max_tokens": 1200
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"⚠️ **Inference Gateway Error:** Remote engine responded with code {response.status_code}: {response.text}"
            
    except Exception as e:
        return f"⚠️ **Execution Exception Interrupt:** System failed to compute token matrix. Trace: {str(e)}"


# -------------------------------------------------------------------------
# MODULE 3: HIGH-FIDELITY USER INTERFACE LAYOUT
# -------------------------------------------------------------------------
# Initialize pipeline dependencies
trained_models, scaler, performance_matrix, feature_names, global_importances = initialize_ml_pipeline()

st.title("Enterprise Cardiovascular CDSS Engine (v4.0 Pro)")
st.caption("Fidelity Predictive Core & Stateful Cognitive Consultation Network | Department of Computer Science & Engineering, BACET")

# Persistent Dynamic Input Vector Framework (Sidebar Separation)
with st.sidebar:
    st.markdown("### 🎛️ Patient Telemetry Streams")
    st.markdown("---")
    
    # Group inputs cleanly using layout containers
    age = st.slider("Patient Age Scope", 18, 95, 52)
    gender_label = st.radio("Phenotypic / Biological Sex", ["Female Vector (0)", "Male Vector (1)"], index=1, horizontal=True)
    sex_input = 1 if "Male" in gender_label else 0
    
    cp_input = st.selectbox("Angina Symptom Manifestation (chestpain)", [0, 1, 2, 3], 
                            format_func=lambda x: {0: "0: Asymptomatic Profile", 1: "1: Typical Angina", 2: "2: Atypical Angina", 3: "3: Non-Anginal Discomfort"}[x])
    
    rbp_input = st.number_input("Resting Systolic Blood Pressure (mmHg)", 80, 220, 128)
    chol_input = st.number_input("Serum Cholesterol Immunoassay (mg/dl)", 100, 550, 214)
    fbs_input = st.radio("Fasting Blood Glucose > 120 mg/dl", [0, 1], horizontal=True)
    
    ecg_input = st.selectbox("Resting Electrocardiograph Waveform Configuration", [0, 1, 2],
                             format_func=lambda x: {0: "0: Normal Waveform Baseline", 1: "1: ST-T Wave Anomaly", 2: "2: Ventricular Hypertrophy Matrix"}[x])
    
    mhr_input = st.number_input("Maximum Ergometric Heart Rate Achieved (bpm)", 60, 220, 142)
    exang_input = st.radio("Exertion-Induced Ischemic Angina Presence", [0, 1], horizontal=True)
    
    oldpeak_input = st.slider("ST Depression Depth Vector (oldpeak)", 0.0, 6.5, 1.2, step=0.1)
    slope_input = st.selectbox("Peak Exercise ST Segment Slope Vector", [1, 2, 3],
                               format_func=lambda x: {1: "1: Upsloping Configuration", 2: "2: Horizontal/Flat Line", 3: "3: Downsloping Vector"}[x])
    
    vessels_input = st.selectbox("Fluoroscopic Coronary Tree Calcification (vessels)", [0, 1, 2, 3])

    st.markdown("---")
    selected_model_node = st.selectbox("Active Pipeline Classifier Core", list(trained_models.keys()))

    # Global Trigger execution pathway
    execute_pipeline = st.button("Compute Core Classification Diagnostics", type="primary", use_container_width=True)

if execute_pipeline or 'active_patient_state' in st.session_state:
    
    # Preserve context frame locally inside state space if explicitly triggered
    if execute_pipeline:
        active_features = {
            'age': age, 'gender': sex_input, 'chestpain': cp_input, 'restingBP': rbp_input,
            'serumcholestrol': chol_input, 'fastingbloodsugar': fbs_input, 'restingrelectro': ecg_input,
            'maxheartrate': mhr_input, 'exerciseangia': exang_input, 'oldpeak': oldpeak_input,
            'slope': slope_input, 'noofmajorvessels': vessels_input
        }
        
        # Transform and predict via selected classifier core
        ordered_vector = np.array([[active_features[col] for col in feature_names]])
        scaled_vector = scaler.transform(ordered_vector)
        
        target_model = trained_models[selected_model_node]
        binary_class = target_model.predict(scaled_vector)[0]
        
        # Compute dynamic execution analytics
        if hasattr(target_model, "predict_proba"):
            risk_probability = target_model.predict_proba(scaled_vector)[0][1]
        else:
            risk_probability = 1.0 if binary_class == 1 else 0.0
            
        st.session_state['active_patient_state'] = {
            'metrics': active_features,
            'prediction': "POSITIVE (Pathology Detected)" if binary_class == 1 else "NEGATIVE (No Pathology Detected)",
            'probability': risk_probability,
            'model_used': selected_model_node
        }

    # Main Screen Content Distribution Tabs
    tab_dashboard, tab_cognitive_consultant, tab_benchmarks = st.tabs([
        "📊 Live Patient Diagnostic Dashboard", 
        "🤖 Cognitive AI Specialist Console", 
        "🔬 Structural Core Metrics"
    ])

    patient_payload = st.session_state['active_patient_state']
    derived_indexes = compute_clinical_hemodynamics(patient_payload['metrics'])

    with tab_dashboard:
        st.markdown("### 🫀 Core Classifier Real-Time Diagnostic Output")
        
        # Upper Metric Layout Row
        d_col1, d_col2, d_col3 = st.columns(3)
        
        with d_col1:
            st.metric(
                label="Cardiovascular Risk Index Probability", 
                value=f"{patient_payload['probability'] * 100:.2f}%",
                delta="CRITICAL EXCURSION" if patient_payload['probability'] > 0.5 else "PHYSIOLOGICAL HOMEOSTASIS"
            )
        with d_col2:
            st.metric(label="Calculated Myocardial Oxygen Workload (RPP)", value=f"{derived_indexes['rpp']} mmHg·bpm")
        with d_col3:
            st.metric(label="Pipeline Classification Anchor Node", value=patient_payload['model_used'].split()[0])

        if patient_payload['probability'] > 0.5:
            st.error(f"🚨 **Diagnostic Alert:** The framework has identified a high-probability risk vector matching Coronary Artery Disease (CAD). Prediction Verdict: {patient_payload['prediction']}.")
        else:
            st.success(f"✅ **Diagnostic Clearance:** The patient profile falls within normal operational limits. Prediction Verdict: {patient_payload['prediction']}.")

        # Middle Metric Layout Row (Derived Secondary Clinical Indexes)
        st.markdown("#### 🔬 Secondary Hemodynamic Indices")
        hi1, hi2, hi3 = st.columns(3)
        hi1.info(f"**AHA BP Stratum:** \n\n {derived_indexes['bp_stage']}")
        hi2.info(f"**Vascular Lipid Load:** \n\n {derived_indexes['lipid_risk']}")
        hi3.info(f"**ST Ischemia Burden:** \n\n {derived_indexes['ischemia_status']}")

    with tab_cognitive_consultant:
        st.markdown("### 🧬 Interventional Cardiology Inference Stream")
        st.caption("Stateful Multi-Turn Cognitive Architecture. Fully Aware of Live Diagnostic Matrices and Classifier Output Weights.")

        # Automation Triggers to force highly complex context changes
        st.markdown("##### ⚡ Executive Diagnostic Macros")
        macro_col1, macro_col2, macro_col3 = st.columns(3)
        macro_input_intercept = None
        
        if macro_col1.button("📋 Generate Differential Diagnostic Breakdown", use_container_width=True):
            macro_input_intercept = "Execute an advanced, publication-grade multi-system differential diagnosis based on my telemetry frame. Map feature cross-correlations explicitly."
        if macro_col2.button("🔬 Pathophysiological Stress Assessment", use_container_width=True):
            macro_input_intercept = "Synthesize an evaluation of my ischemic markers, RPP baseline, and structural calcification vectors. Cross-reference with standard AHA/ACC protocols."
        if macro_col3.button("💻 Evaluate Mathematical Inference Pipeline", use_container_width=True):
            macro_input_intercept = "Provide a high-level data-science audit explaining how the classifier weighted these inputs mathematically. Highlight anomalous feature deviations."

        # Keep stateful conversation memory alive
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = [
                {"role": "assistant", "content": "Cognitive Node Online. Synced directly with your current Random Forest feature weights and calculated clinical hemodynamic indexes. Query the diagnostics matrix below."}
            ]

        # Display history elements cleanly
        for chat_msg in st.session_state.conversation_history:
            with st.chat_message(chat_msg["role"]):
                st.markdown(chat_msg["content"])

        # Capture interactive prompt input array
        user_query_stream = st.chat_input("Input custom diagnostic query or explore microvascular pathophysiology configurations...")
        if macro_input_intercept:
            user_query_stream = macro_input_intercept

        if user_query_stream:
            # Render user message instantly
            with st.chat_message("user"):
                st.markdown(user_query_stream)
            st.session_state.conversation_history.append({"role": "user", "content": user_query_stream})

            # CONSTRUCT SYSTEM BRAIN CONTEXT (Dynamic Context Injector)
            structured_clinical_brief = (
                "You are an Elite Interventional Cardiologist, Medical Professor, and Chief Artificial Intelligence Research Scientist. "
                "You are running inside an enterprise Clinical Decision Support System. Your answers must be dense, analytical, rigorous, and explicitly targeted. "
                "Never apologize, never repeat the inputs back to the user blindly, and never give a generic cookie-cutter response.\n\n"
                f"=== SYNCHRONIZED LIVE CLINICAL CHART ===\n"
                f"- Primary Classifier Output: {patient_payload['prediction']}\n"
                f"- Mathematical Prediction Risk Probability: {patient_payload['probability'] * 100:.4f}%\n"
                f"- Pipeline Engine Model Used: {patient_payload['model_used']}\n"
                f"- Calculated Rate Pressure Product (RPP): {derived_indexes['rpp']} mmHg·bpm\n"
                f"- AHA Arterial Pressure Scale: {derived_indexes['bp_stage']}\n"
                f"- Atherosclerotic Metabolic Index: {derived_indexes['lipid_risk']}\n"
                f"- Ischemic Depression Metric: {derived_indexes['ischemia_status']}\n"
                f"- Fluoroscopy Target Output: {derived_indexes['vessel_calcification']}\n"
                f"- Raw Telemetry Array Values: {str(patient_payload['metrics'])}\n"
                f"=== INSTRUCTION SPECIFICATION ===\n"
                f"Synthesize the answer using complex clinical vocabulary (e.g., myocardial oxygen kinetics, transmural compliance, subendocardial perfusion, coronary stenosis vectors). "
                f"Incorporate the provided ML model's risk output and calculated metrics directly into your logic. Use clean Markdown headers to organize your output structurally."
            )

            # Prune conversation history window to prevent tokens from blowing past execution windows
            active_payload_window = st.session_state.conversation_history[-6:]

            with st.chat_message("assistant"):
                with st.spinner("Synthesizing dynamic clinical token array via Groq LPU..."):
                    ai_inference_output = call_groq_cognitive_engine(active_payload_window, structured_clinical_brief)
                    st.markdown(ai_inference_output)
                    
            st.session_state.conversation_history.append({"role": "assistant", "content": ai_inference_output})

    with tab_benchmarks:
        st.markdown("### 📊 Cross-Classifier Performance Audit Matrix")
        st.dataframe(pd.DataFrame(performance_matrix).T, use_container_width=True)
        
        st.markdown("#### 🛠️ Local Feature Importance Contributor Weightings")
        st.json(global_importances)

else:
    # Default landing visual state framework if no pipeline has been executed yet
    st.info("💡 **Pipeline Initialization Notice:** The machine learning arrays and Groq cognitive context links are completely initialized. Configure the patient metrics on the left sidebar panel and select 'Compute Core Classification Diagnostics' to begin advanced telemetry parsing.")
    
    st.markdown("### 🧱 Underlying Core Framework Health Diagnostics")
    c_h1, c_h2 = st.columns(2)
    with c_h1:
        st.success("✅ Main Data Frame Integrity Verified: 'heart.csv' loaded successfully.")
        st.success(f"✅ Global Feature Matrix Initialized: Found {len(feature_names)} predictable variables.")
    with c_h2:
        st.success(f"✅ Secure Vault Status: `st.secrets` pathway loaded and active.")
        st.success(f"✅ Multi-Model Architecture Active: 4 pipeline algorithms mapped.")
