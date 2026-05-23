import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Configure the Streamlit page
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Loads pre-trained model artifacts defensively."""
    required_files = ['best_model.pkl', 'scaler.pkl', 'results.pkl']
    
    # Check if artifacts exist
    if not all(os.path.exists(f) for f in required_files):
        st.error("⚠️ System Offline: Model artifacts not found. Please run the training pipeline first.")
        st.stop()
        
    try:
        with open('best_model.pkl', 'rb') as f: model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
        with open('results.pkl', 'rb') as f: results = pickle.load(f)
        return model, scaler, results
    except Exception as e:
        st.error(f"⚠️ System Offline: Artifact corruption detected. Error: {e}")
        st.stop()

# Initialize UI
st.title("Heart Disease Prediction System")
st.caption("Developed by Arpan, Chandan & MD Belal | Indian Cardiovascular Dataset")

# Load system artifacts
model, scaler, results = load_artifacts()

# Build Application Tabs
tab1, tab2, tab3 = st.tabs(["Predict", "Performance", "About"])

with tab1:
    st.write("### Input Clinical Parameters")
    
    # UI Layout
    c1, c2, c3 = st.columns(3)
    
    with c1:
        age = st.slider("Age", 20, 80, 50)
        gender = st.radio("Gender", ["Female (0)", "Male (1)"], index=1, horizontal=True)
        gender_val = 1 if "Male" in gender else 0
        chestpain = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
        restingrelectro = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
        
    with c2:
        resting_bp = st.number_input("Resting BP (mm Hg)", 90, 200, 120)
        serumcholestrol = st.number_input("Serum Cholesterol (mg/dl)", 120, 600, 200)
        fastingbloodsugar = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1], horizontal=True)
        slope = st.selectbox("ST Slope (1-3)", [1, 2, 3])
        
    with c3:
        maxheartrate = st.number_input("Max Heart Rate", 70, 210, 150)
        exerciseangia = st.radio("Exercise Induced Angina", [0, 1], horizontal=True)
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.2, 1.0, step=0.1)
        noofmajorvessels = st.selectbox("Major Vessels Colored by Flourosopy (0-3)", [0, 1, 2, 3])
    
    st.markdown("---")
    
    # Inference Trigger
    if st.button("Predict Result", type="primary", use_container_width=True):
        
        # 1. Positional Feature Mapping (Resilient against column name changes)
        # Order MUST match the 12 clinical attributes in the training pipeline
        raw_features = [
            age, 
            gender_val, 
            chestpain, 
            resting_bp, 
            serumcholestrol, 
            fastingbloodsugar, 
            restingrelectro, 
            maxheartrate, 
            exerciseangia, 
            oldpeak, 
            slope, 
            noofmajorvessels
        ]
        
        try:
            # 2. Vectorization
            feature_vector = np.array(raw_features).reshape(1, -1)
            
            # 3. Shape Validation (Defensive Check)
            if feature_vector.shape[1] != 12:
                st.error(f"System Error: Expected 12 input features, received {feature_vector.shape[1]}.")
                st.stop()
                
            # 4. Scaling & Inference
            scaled_input = scaler.transform(feature_vector) 
            prediction = model.predict(scaled_input)[0]
            
            # 5. Output Routing
            if prediction == 1: 
                st.error("⚠️ **Diagnosis:** Heart Disease Detected. Please consult a cardiologist.")
            else: 
                st.success("✅ **Diagnosis:** No Heart Disease Detected. Keep up the healthy lifestyle!")
                
        except Exception as e:
            st.error(f"Inference Engine Failure: {str(e)}")
            st.info("Please verify the integrity of your scaler.pkl file.")

with tab2:
    st.write("### Model Performance Metrics")
    try:
        # Dynamically render the performance of all trained models
        df_res = pd.DataFrame(results).T.reset_index()
        df_res.columns = ['Model Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
        st.dataframe(df_res.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
        
        # Identify the active model
        best_algo = max(results, key=lambda k: results[k]['Accuracy'])
        st.caption(f"Currently active production model: **{best_algo}**")
    except Exception as e:
        st.warning("Performance metrics currently unavailable. Run the training pipeline to generate `results.pkl`.")

with tab3:
    st.write("### System Architecture & Background")
    st.write("""
    This inference engine is built on the **Indian Cardiovascular Disease Dataset (Mendeley)**. 
    It features a robust 12-parameter predictive pipeline designed to localize diagnosis logic 
    for South Asian demographics.
    
    **Developed at BACET by:**
    * Arpan Das
    * Chandan Kumar Mishra
    * MD Belal
    
    *System Status: Active | Resilient Pipeline v2.0*
    """)
