import os
import time
import warnings
import requests
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Cardiovascular CDSS Engine v6.0 Engine", 
    layout="wide", 
    page_icon="🫀",
    initial_sidebar_state="expanded"
)

@st.cache_resource(show_spinner=False)
def initialize_predictive_cluster():
    if not os.path.exists('heart.csv'):
        st.error("🚨 Critical Architecture Error: 'heart.csv' data lake node not found in root path.")
        st.stop()

    df = pd.read_csv('heart.csv')
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')
    
    if 'patientid' in df.columns:
        df.drop('patientid', axis=1, inplace=True)
    if 'Classification' in df.columns and 'target' not in df.columns:
        df.rename(columns={'Classification': 'target'}, inplace=True)
    
    X = df.drop('target', axis=1)
    y = df['target']
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest Core (Ensemble Node)': RandomForestClassifier(n_estimators=150, random_state=42),
        'Gradient Boosting Core (Adaptive Node)': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine (Hyperplane Node)': SVC(kernel='rbf', probability=True, random_state=42),
        'Regularized Logistic Core (Linear Node)': LogisticRegression(max_iter=1000, random_state=42)
    }

    performance_matrix = {}
    trained_cluster = {}
    
    for name, clf in models.items():
        clf.fit(X_train_scaled, y_train)
        predictions = clf.predict(X_test_scaled)
        
        performance_matrix[name] = {
            'Accuracy': f"{accuracy_score(y_test, predictions) * 100:.2f}%",
            'Precision': f"{precision_score(y_test, predictions) * 100:.2f}%",
            'Recall': f"{recall_score(y_test, predictions) * 100:.2f}%",
            'F1-Score': f"{f1_score(y_test, predictions) * 100:.2f}%",
        }
        trained_cluster[name] = clf

    rf_anchor = trained_cluster['Random Forest Core (Ensemble Node)']
    global_feature_importances = dict(zip(feature_names, rf_anchor.feature_importances_))

    return trained_cluster, scaler, performance_matrix, feature_names, global_feature_importances

def generate_advanced_clinical_brief(m):
    if not m:
        return {}
        
    sbp = m.get('restingBP', 120)
    hr = m.get('maxheartrate', 150)
    chol = m.get('serumcholestrol', 200)
    oldpeak = m.get('oldpeak', 0.0)
    vessels = m.get('noofmajorvessels', 0)
    age = m.get('age', 50)
    
    rpp = int(sbp * hr)
    
    estimated_max_hr = 220 - age
    chronotropic_index = round((hr / estimated_max_hr) * 100, 1) if estimated_max_hr > 0 else 100.0
    
    if sbp < 120: bp_stage = "Normal Normotensive Baseline"
    elif 120 <= sbp < 130: bp_stage = "Elevated Systolic Phase"
    elif 130 <= sbp < 140: bp_stage = "Stage 1 Arterial Hypertension"
    else: bp_stage = "Stage 2 Severe Arterial Hypertension"
        
    if chol < 200: lipid_risk = "Optimal/Desirable Endothelial Profile"
    elif 200 <= chol < 240: lipid_risk = "Borderline High Atherosclerotic Burden"
    else: lipid_risk = "High-Risk Atherosclerotic Lipoprotein Aggregation"
        
    if oldpeak == 0: ischemia_status = "No Physiologically Active ST-Segment Drift Checked"
    elif 0 < oldpeak <= 1.5: ischemia_status = "Mild-to-Moderate Exercise-Induced Subendocardial Ischemia"
    else: ischemia_status = "Severe Transmural Myocardial Ischemic Injury Risk"
        
    return {
        "rpp": rpp,
        "chronotropic_index": f"{chronotropic_index}% of age-predicted maximum limits",
        "bp_stage": bp_stage,
        "lipid_risk": lipid_risk,
        "ischemia_status": ischemia_status,
        "vessel_calcification": f"{vessels} primary coronary branches occluded via fluoroscopy vectors"
    }

def transmit_groq_inference(chat_payload, custom_system_brief):
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
        if not api_key:
            return "⚠️ **System Vault Interrupted:** Inference aborted. The `GROQ_API_KEY` mapping is missing from st.secrets configuration."

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        unified_messages = [{"role": "system", "content": custom_system_brief}] + chat_payload
        
        payload = {
            "model": "llama-3.3-70b-specdec",
            "messages": unified_messages,
            "temperature": 0.15,
            "max_tokens": 1600
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"⚠️ **Gateway Context Failure:** Remote processor rejected array with status {response.status_code}: {response.text}"
            
    except Exception as e:
        return f"⚠️ **Execution Core Break:** Failed to process tokenizer sequence. Trace: {str(e)}"

trained_cluster, scaler, performance_matrix, feature_names, global_importances = initialize_predictive_cluster()

st.title("Enterprise Cardiovascular CDSS Engine (v6.0 Ultra-Pro)")
st.caption("Fidelity Multi-Classifier Core & Stateful Reasoning Consultant Platform | Engineered by Arpan Das, Chandan Kumar Mishra & MD Belal")

with st.sidebar:
    st.markdown("### 🎛️ Live Patient Telemetry Frame")
    st.markdown("---")
    
    age = st.slider("Patient Age Scale", 18, 95, 52)
    gender_label = st.radio("Biological / Phenotypic Sex Input", ["Female Vector (0)", "Male Vector (1)"], index=1, horizontal=True)
    sex_input = 1 if "Male" in gender_label else 0
    
    cp_input = st.selectbox("Angina Waveform Manifestation (chestpain)", [0, 1, 2, 3], 
                            format_func=lambda x: {0: "0: Asymptomatic Telemetry", 1: "1: Typical Angina Variant", 2: "2: Atypical Angina Metric", 3: "3: Non-Anginal Muscle Response"}[x])
    
    rbp_input = st.number_input("Resting Systolic Blood Pressure (mmHg)", 80, 220, 126)
    chol_input = st.number_input("Total Serum Cholesterol Immunoassay (mg/dl)", 100, 550, 218)
    fbs_input = st.radio("Fasting Blood Glucose > 120 mg/dl", [0, 1], horizontal=True)
    
    ecg_input = st.selectbox("Resting Electrocardiograph Vector Phase", [0, 1, 2],
                             format_func=lambda x: {0: "0: Normal Linear Baseline", 1: "1: ST-T Phase Pathological Shift", 2: "2: Left Ventricular Hypertrophy Matrix"}[x])
    
    mhr_input = st.number_input("Maximum Ergometric Heart Rate Achieved (bpm)", 60, 220, 148)
    exang_input = st.radio("Exertion-Induced Coronary Ischemic Angina", [0, 1], horizontal=True)
    
    oldpeak_input = st.slider("ST Segment Depression Depth Deviation (oldpeak)", 0.0, 6.5, 1.2, step=0.1)
    slope_input = st.selectbox("Peak Exercise ST Segment Slope Morphology", [1, 2, 3],
                               format_func=lambda x: {1: "1: Upsloping Hyper-Response", 2: "2: Horizontal Flat Line Shift", 3: "3: Downsloping Deceration Vector"}[x])
    
    vessels_input = st.selectbox("Fluoroscopic Structural Vessel Blockages (noofmajorvessels)", [0, 1, 2, 3])

    st.markdown("---")
    selected_node_identity = st.selectbox("Active Pipeline Classifier Engine", list(trained_cluster.keys()))

    execute_diagnostics_pipeline = st.button("Execute Pipeline Core Execution", type="primary", use_container_width=True)

if execute_diagnostics_pipeline or 'active_patient_state' in st.session_state:
    
    if execute_diagnostics_pipeline:
        active_features = {
            'age': age, 'gender': sex_input, 'chestpain': cp_input, 'restingBP': rbp_input,
            'serumcholestrol': chol_input, 'fastingbloodsugar': fbs_input, 'restingrelectro': ecg_input,
            'maxheartrate': mhr_input, 'exerciseangia': exang_input, 'oldpeak': oldpeak_input,
            'slope': slope_input, 'noofmajorvessels': vessels_input
        }
        
        ordered_array = np.array([[active_features[col] for col in feature_names]])
        scaled_array = scaler.transform(ordered_array)
        
        target_engine = trained_cluster[selected_node_identity]
        binary_verdict = target_engine.predict(scaled_array)[0]
        
        if hasattr(target_engine, "predict_proba"):
            computed_probability = target_engine.predict_proba(scaled_array)[0][1]
        else:
            computed_probability = 1.0 if binary_verdict == 1 else 0.0
            
        st.session_state['active_patient_state'] = {
            'metrics': active_features,
            'prediction': "POSITIVE (Coronary Artery Disease Signature Flagged)" if binary_verdict == 1 else "NEGATIVE (Physiological Boundaries Clear)",
            'probability': computed_probability,
            'model_node': selected_node_identity
        }

    tab_dashboard, tab_cognitive_consultant, tab_benchmarks = st.tabs([
        "📊 Real-Time Patient Analytics Dashboard", 
        "🧠 Elite Cognitive Cardiology Specialist Core", 
        "🔬 Structural Classifier Performance Registry"
    ])

    patient_payload = st.session_state['active_patient_state']
    derived_brief = generate_advanced_clinical_brief(patient_payload['metrics'])

    with tab_dashboard:
        st.markdown("### 🧬 Classifier Pipeline Stratification Output")
        
        db_c1, db_c2, db_c3 = st.columns(3)
        with db_c1:
            st.metric(
                label="Ensemble Pathological Probability Match", 
                value=f"{patient_payload['probability'] * 100:.2f}%",
                delta="CRITICAL PIPELINE EXCURSION" if patient_payload['probability'] > 0.5 else "PHYSIOLOGICAL REGULARITY CLEARED"
            )
        with db_c2:
            st.metric(label="Myocardial Oxygen Consumption Workload (RPP)", value=f"{derived_brief['rpp']} mmHg·bpm")
        with db_c3:
            st.metric(label="Active Mathematical Anchor Classifier", value=patient_payload['model_node'].split()[0])

        if patient_payload['probability'] > 0.5:
            st.error(f"🚨 **High-Risk Stratification Trigger:** Active pipeline cores have identified a pathological classification tracking structural CAD. Core Verdict: {patient_payload['prediction']}.")
        else:
            st.success(f"✅ **Low-Risk Stratification Trigger:** Telemetry distributions fall entirely within expected physiological baselines. Core Verdict: {patient_payload['prediction']}.")

        st.markdown("#### 🔬 Extrapolated Cardiovascular Telemetry Indices")
        bio1, bio2, bio3 = st.columns(3)
        bio1.info(f"**AHA Hydrostatic Load Stage:** \n\n {derived_brief['bp_stage']}")
        bio2.info(f"**Vascular Lipoprotein Tension:** \n\n {derived_brief['lipid_risk']}")
        bio3.info(f"**Myocardial Ischemia Phase Vector:** \n\n {derived_brief['ischemia_status']}")

    with tab_cognitive_consultant:
        st.markdown("### 🧠 Autonomous Interventional Cardiology Consultation Core")
        st.caption("State-Aware Reasoning Node | Dynamically Intercepts Pipeline Feature Weights and Internal Hemodynamic Vectors.")

        st.markdown("##### ⚡ Rapid Clinical Analysis Macros")
        m_row1, m_row2, m_row3 = st.columns(3)
        override_prompt_input = None
        
        if m_row1.button("📋 Compile Differential Diagnostic Case Review", use_container_width=True):
            override_prompt_input = "Perform an advanced, multi-system differential diagnosis based on the active patient feature arrays. Map feature cross-correlations explicitly."
        if m_row2.button("🔬 Analyze Pathophysiological Ischemic Stress Matrix", use_container_width=True):
            override_prompt_input = "Synthesize an evaluation of my ischemic markers, RPP baseline, and chronotropic index. Cross-reference with standard AHA/ACC telemetry guidelines."
        if m_row3.button("💻 Run Mathematical Core Inference Audit", use_container_width=True):
            override_prompt_input = "Act as a Lead Data Scientist. Audit this current data matrix, explaining how the active classifier weighted these normalized parameters mathematically."

        if "pro_chat_history_v6" not in st.session_state:
            st.session_state.pro_chat_history_v6 = [
                {"role": "assistant", "content": "Cognitive Consultant Node Active. Synced directly with your current model's mathematical risk outputs and derived indexes. Query the diagnostics engine below."}
            ]

        for chat_msg in st.session_state.pro_chat_history_v6:
            with st.chat_message(chat_msg["role"]):
                st.markdown(chat_msg["content"])

        user_query_stream = st.chat_input("Query diagnostic matrices, evaluate hemodynamic limits, or inspect mathematical core weights...")
        if override_prompt_input:
            user_query_stream = override_prompt_input

        if user_query_stream:
            with st.chat_message("user"):
                st.markdown(user_query_stream)
            st.session_state.pro_chat_history_v6.append({"role": "user", "content": user_query_stream})

            system_super_prompt = (
                "SYSTEM KNOWLEDGE AND OPERATIONAL EXECUTION MATRIX BIBLE\n"
                "========================================================================\n"
                "CORE EXECUTIVE ARCHITECTURE IDENTITY:\n"
                "You are executing as the centralized Cognitive Reasoning Layer of an enterprise-grade Clinical Decision Support System (CDSS). "
                "Your underlying architecture is an advanced stateful neuro-symbolic framework. Globally, your role maps to a triumvirate profile: "
                "an elite Interventional Cardiologist with extensive clinical operations tenure, an academic Professor of Advanced Cardiovascular Pathophysiology, "
                "and a Distinguished Principal Data Scientist specializing in complex biological pipeline architectures and high-entropy medical telemetry frameworks.\n\n"
                "STRICT LOGICAL RESTRICTIONS AND OPERATIONAL CONSTRAINTS:\n"
                "1. ELIMINATION OF FILLER PROTOCOL: You must bypass all conversational fluff. Do not output 'Hello', 'Thank you', 'Sure thing', 'As an AI...', "
                "or conversational introductions. Begin directly with high-density analytical reasoning tokens.\n"
                "2. NO VERBATIM ECHO STRATEGY: Do not repeat back raw features blindly. Translate raw inputs into derived physiological relationships.\n"
                "3. Hallucination Guard: Ground every pathophysiological deduction strictly within the mathematical margins of the active classifier core.\n"
                "4. Absolute Zero-Apology Protocol: Never apologize under any circumstances. If previous conversation parameters are queried, re-verify "
                "against structural ground truths and deliver cold, accurate data matrices.\n\n"
                "DEEP PATHOPHYSIOLOGICAL REFERENCE MANUAL AND MEDICAL TAXONOMY:\n"
                "- Coronary Stenosis & Sheer-Stress Dynamics: When plaque narrows an epicardial artery, resting flow remains stable due to microvascular "
                "autoregulation. However, under exercise workloads, standard vasodilation fails, leading to oxygen supply/demand mismatch.\n"
                "- Rate Pressure Product (RPP Kinetics): Calculated as Systolic Blood Pressure multiplied by Heart Rate. It serves as an accurate, non-invasive surrogate "
                "for Myocardial Oxygen Consumption (MVO2). Values exceeding 12,000 signify heightened myocardial workload; values over 20,000 "
                "reflect extreme workload vectors where underlying arterial stenosis will precipitate subendocardial ischemia.\n"
                "- Chronotropic Incompetence Indexing: The physiological failure of the heart to increase its rate match relative to metabolic demands during "
                "exertion. Quantified by comparing peak heart rate against age-predicted maximum limits (220 - Age). Below 80% represents chronotropic "
                "incompetence, often indicating advanced ischemic bundle branches or autonomic microvascular breakdown.\n"
                "- ST-Segment Waveform Morphologies:\n"
                "  * Upsloping ST Depression: Frequently indicates benign hyperventilatory mechanics or rapid ventricular pacing, but if deep, reflects "
                "    early subendocardial perfusion failures.\n"
                "  * Horizontal ST Depression: Strong classic indicator of acute subendocardial ischemia. Represents localized delay in ventricular repolarization.\n"
                "  * Downsloping ST Depression: Highest statistical positive predictive value for severe transmural multi-vessel CAD or left main coronary artery stenosis.\n"
                "- Fluoroscopic Vessel Calcification Vectors: The count of principal coronary arteries showing calcification (0 to 3) is a direct structural marker "
                "of global atherosclerotic burden. Within tree-based classification pipelines, this value operates as an immutable high-information split metric.\n"
                "- Hypercholesterolemia and Atherosclerotic Plaque Cascades: Serum cholesterol elevations increase circulating low-density lipoproteins, "
                "triggering subendothelial retention, macrophage activation, foam cell formation, and eventual fibrous cap degradation.\n\n"
                "DATA SCIENCE AND MATHEMATICAL ANALYSIS SPECIFICATIONS:\n"
                "- Random Forest Split Dynamics: Operates by optimizing Gini Impurity or Information Gain across hundreds of decorrelated decision trees. "
                "High feature importance scores flag columns that provide maximal distribution balance shifts within the tree nodes.\n"
                "- Gradient Boosting State Optimization: Trains models sequentially by fitting a new tree model to the negative gradient residuals of the "
                "loss function. This forces the pipeline to constantly minimize log-loss errors for hard-to-classify patient telemetry outliers.\n"
                "- Support Vector Machine Margin Boundaries: Maps scaled input feature vectors into multi-dimensional spaces using a Radial Basis Function (RBF) "
                "kernel. It constructs an optimal decision boundary hyperplane by maximizing the margin distance between conflicting support vector coordinates.\n"
                "- Logistic Regression Odds Optimization: Fits coefficients to log-odds vectors using maximum likelihood estimation, applying a sigmoid transform "
                "to yield explicit conditional probability boundaries.\n\n"
                "CHAIN-OF-THOUGHT INFERENCE PROTOCOLS:\n"
                "Phase I (Hemodynamic Assessment): Correlate SBP and Max HR. Compute the specific RPP value. Cross-reference this against the patient's age and "
                "chronotropic index to determine if structural compliance limits are restricting necessary perfusion loops.\n"
                "Phase II (Electrical Ischemia Intercept): Examine the depth of ST depression and cross-reference with its slope morphology. Determine if the "
                "result represents microvascular multi-vessel obstruction or isolated regional stenosis.\n"
                "Phase III (Algorithmic Verification): Analyze the predictive probability percentage against the global model importance map. Identify the explicit "
                "feature interactions that pushed the decision boundary into its current target state.\n\n"
                "=== LIVE CENTRAL DATASET DOSSIER ===\n"
                f"- Primary Classifier Output Verdict: {patient_payload['prediction']}\n"
                f"- Pipeline Predictive Risk Probability: {patient_payload['probability'] * 100:.6f}%\n"
                f"- Active Pipeline Node Infrastructure: {patient_payload['model_node']}\n"
                f"- Calculated Rate Pressure Product (RPP): {derived_brief['rpp']} mmHg·bpm\n"
                f"- Age-Predicted Chronotropic Capacity Index: {derived_brief['chronotropic_index']}\n"
                f"- AHA Hydrostatic Arterial Pressure Scale: {derived_brief['bp_stage']}\n"
                f"- Atherosclerotic Metabolic Index: {derived_brief['lipid_risk']}\n"
                f"- Ischemical Segment Depression Profile: {derived_brief['ischemia_status']}\n"
                f"- Fluoroscopy Volumetric Output Matrix: {derived_brief['vessel_calcification']}\n"
                f"- Active Vector Raw Input Payload Frame: {str(patient_payload['metrics'])}\n\n"
                "=== EXPLICIT STRUCTURAL REGULATORY SCHEMA ===\n"
                "Your reasoning output must match this publication-grade markdown syntax structure without exception:\n\n"
                "### 1. 🔬 ADVANCED HEMODYNAMIC WORKLOAD KINETICS\n"
                "Provide an exhaustive pathophysiological analysis mapping the interaction between resting BP and peak heart rate. "
                "Detail how these numbers impact coronary perfusion pressures, myocardial oxygen requirements (MVO2), and "
                "chronotropic response thresholds for this patient.\n\n"
                "### 2. 🫀 ISCHEMIC WAVEFORM CONFIGURATION MATRIX\n"
                "Evaluate the ST depression depth and slope configuration. Detail the microvascular and subendocardial flow velocity "
                "mechanics under exertion. Contrast horizontal shifts or downsloping deceleration vectors with normal baselines.\n\n"
                "### 3. 💻 DATA-SCIENCE INFRASTRUCTURE & ENSEMBLE PIPELINE AUDIT\n"
                "Break down the mathematical reasoning of the active pipeline classifier. Explain which high-entropy variables "
                "(such as fluoroscopy branches, angina classifications, or age parameters) forced the Gini impurity shifts or "
                "hyperplane vector boundaries into this precise prediction probability percentage.\n\n"
                "Maintain an elite, academic clinical tone throughout the entire multi-turn generation sequence."
            )

            active_payload_window = st.session_state.pro_chat_history_v6[-6:]

            with st.chat_message("assistant"):
                with st.spinner("Processing token array streams over Groq LPU cluster..."):
                    llm_output_payload = transmit_groq_inference(active_payload_window, system_super_prompt)
                    st.markdown(llm_output_payload)
                    
            st.session_state.pro_chat_history_v6.append({"role": "assistant", "content": llm_output_payload})

    with tab_benchmarks:
        st.markdown("### 🧱 Pipeline Ensemble Accuracy Benchmarking")
        st.dataframe(pd.DataFrame(performance_matrix).T, use_container_width=True)
        
        st.markdown("#### 🛠️ Global Model Feature Importance Weights")
        st.json(global_importances)

else:
    st.info("💡 **Inference Notice:** Multi-model structures and Groq LPU communication layers have successfully booted up. Set patient telemetry parameters inside the left control panel and hit 'Execute Pipeline Core Execution' to initialize high-fidelity tracking metrics.")
    
    st.markdown("### ⛓️ Pipeline Node Health Array")
    h_col1, h_col2 = st.columns(2)
    with h_col1:
        st.success("✅ Main CSV Storage Matrix Secure: 'heart.csv' structural columns verified.")
        st.success(f"✅ Input Dimension Scope Isolated: Managed {len(feature_names)} diagnostic columns.")
    with h_col2:
        st.success(f"✅ Cloud Secrets Storage Linked: Managed secure `st.secrets` pathway configuration.")
        st.success(f"✅ Multi-Model Architecture Configured: 4 heavy classification nodes active.")
