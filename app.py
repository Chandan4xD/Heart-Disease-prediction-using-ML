import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

@st.cache_resource(show_spinner="Initializing ML Engine...")
def initialize_system():
    try:
        if all(os.path.exists(f) for f in ['best_model.pkl', 'scaler.pkl', 'results.pkl']):
            with open('best_model.pkl', 'rb') as f: model = pickle.load(f)
            with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
            with open('results.pkl', 'rb') as f: results = pickle.load(f)
            return model, scaler, results
    except Exception:
        pass

    if not os.path.exists('heart.csv'):
        st.error("Critical Error: 'heart.csv' missing from repository.")
        st.stop()
        
    df = pd.read_csv('heart.csv')
    df.columns = df.columns.str.strip()
    
    if 'patientid' in df.columns:
        df.drop('patientid', axis=1, inplace=True)
    target_col = df.columns[-1]
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    if X.shape[1] != 12:
        st.error(f"Schema Error: Expected 12 features, found {X.shape[1]}.")
        st.stop()
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    for name, m in models.items():
        m.fit(X_train_s, y_train)
        preds = m.predict(X_test_s)
        results[name] = {
            'Accuracy': round(accuracy_score(y_test, preds) * 100, 2),
            'Precision': round(precision_score(y_test, preds) * 100, 2),
            'Recall': round(recall_score(y_test, preds) * 100, 2),
            'F1-Score': round(f1_score(y_test, preds) * 100, 2)
        }
        trained_models[name] = m
        
    best_algo = max(results, key=lambda k: results[k]['Accuracy'])
    return trained_models[best_algo], scaler, results

st.title("Heart Disease Prediction System")
st.caption("Developed by Arpan, Chandan & MD Belal")

model, scaler, results = initialize_system()

tab1, tab2, tab3 = st.tabs(["Predict", "Performance", "About"])

with tab1:
    st.write("### Input Clinical Parameters")
    
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
    
    if st.button("Predict Result", type="primary", use_container_width=True):
        raw_features = [
            age, gender_val, chestpain, resting_bp, serumcholestrol, 
            fastingbloodsugar, restingrelectro, maxheartrate, 
            exerciseangia, oldpeak, slope, noofmajorvessels
        ]
        
        try:
            feature_vector = np.array(raw_features).reshape(1, -1)
            
            if feature_vector.shape[1] != 12:
                st.error(f"System Error: Expected 12 features, received {feature_vector.shape[1]}.")
                st.stop()
                
            scaled_input = scaler.transform(feature_vector) 
            prediction = model.predict(scaled_input)[0]
            
            if prediction == 1: 
                st.error("⚠️ **Diagnosis:** Heart Disease Detected. Please consult a cardiologist.")
            else: 
                st.success("✅ **Diagnosis:** No Heart Disease Detected. Keep up the healthy lifestyle!")
                
        except Exception as e:
            st.error(f"Inference Engine Failure: {str(e)}")

with tab2:
    st.write("### Model Performance Metrics")
    try:
        df_res = pd.DataFrame(results).T.reset_index()
        df_res.columns = ['Model Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
        st.dataframe(df_res.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
        
best_algo = max(results, key=lambda k: (results[k]['Accuracy'], results[k]['F1-Score']))
        st.info(f"🏆 Currently active production model: **{best_algo}**")
    except Exception:
        st.warning("Performance metrics currently unavailable.")

with tab3:
    st.write("### System Architecture & Background")
    st.write("""
    This inference engine is built on the **Indian Cardiovascular Disease Dataset (Mendeley)**. 
    It features a robust 12-parameter predictive pipeline with auto-healing cloud deployment capabilities.
    
    **Developed at BACET by:**
    * Arpan Das
    * Chandan Kumar Mishra
    * MD Belal
    
    *System Status: Active | Resilient Pipeline v4.0 (Cloud Native)*
    """)
