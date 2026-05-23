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

st.set_page_config(page_title="Cardiovascular Diagnosis Engine", layout="wide", page_icon="🫀")

@st.cache_resource(show_spinner="Booting ML Engine...")
def initialize_system():
    artifacts = ['best_model.pkl', 'scaler.pkl', 'results.pkl']
    if all(os.path.exists(f) for f in artifacts):
        try:
            with open('best_model.pkl', 'rb') as f: model = pickle.load(f)
            with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
            with open('results.pkl', 'rb') as f: results = pickle.load(f)
            return model, scaler, results
        except Exception: pass

    if not os.path.exists('heart.csv'):
        st.error("System Failure: 'heart.csv' missing from repository.")
        st.stop()

    df = pd.read_csv('heart.csv')
    df.columns = df.columns.str.strip()

    if 'patientid' in df.columns.str.lower():
        df = df.drop(columns=[df.columns[df.columns.str.lower() == 'patientid'][0]])

    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    X = X.fillna(X.median())
    y = df[target_col].fillna(df[target_col].mode()[0])

    if X.shape[1] != 12:
        st.error(f"Schema Violation: Expected 12 features, detected {X.shape[1]}.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
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

    priority = {'Random Forest': 5, 'Logistic Regression': 4, 'SVM': 3, 'Decision Tree': 2, 'KNN': 1}
    best_algo = max(results, key=lambda k: (float(results[k]['Accuracy']), float(results[k]['F1-Score']), priority[k]))

    return trained_models[best_algo], scaler, results


st.title("Cardiovascular Diagnosis Engine")
st.caption("Core Architecture by Arpan, Chandan & MD Belal")

model, scaler, results = initialize_system()

t1, t2, t3 = st.tabs(["Clinical Assessment", "Engine Diagnostics", "System Overview"])

with t1:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("##### Demographics & Vitals")
        age = st.slider("Age", 20, 80, 50)
        gender = st.radio("Gender", ["Female", "Male"], index=1, horizontal=True)
        gender_val = 1 if gender == "Male" else 0
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)
        maxheartrate = st.number_input("Maximum Heart Rate", 70, 210, 150)

    with c2:
        st.markdown("##### Pain & Cardiac Markers")
        chestpain = st.selectbox("Chest Pain Severity (0-3)", [0, 1, 2, 3])
        exerciseangia = st.radio("Exercise Induced Angina", ["No", "Yes"], horizontal=True)
        exang_val = 1 if exerciseangia == "Yes" else 0
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.2, 1.0, step=0.1)
        slope = st.selectbox("ST Slope (1-3)", [1, 2, 3])

    with c3:
        st.markdown("##### Blood Work & Imaging")
        serumcholestrol = st.number_input("Serum Cholesterol (mg/dl)", 120, 600, 200)
        fastingbloodsugar = st.radio("Fasting Blood Sugar > 120 mg/dl", ["False", "True"], horizontal=True)
        fbs_val = 1 if fastingbloodsugar == "True" else 0
        restingrelectro = st.selectbox("Resting ECG Result (0-2)", [0, 1, 2])
        noofmajorvessels = st.selectbox("Major Vessels Colored (0-3)", [0, 1, 2, 3])

    st.markdown("---")
    
    if st.button("Initialize Scan", type="primary", use_container_width=True):
        raw_features = [
            age, gender_val, chestpain, resting_bp, serumcholestrol,
            fbs_val, restingrelectro, maxheartrate, exang_val, oldpeak, 
            slope, noofmajorvessels
        ]
        try:
            vector = np.array(raw_features).reshape(1, -1)
            prediction = model.predict(scaler.transform(vector))[0]
            if prediction == 1:
                st.error("🚨 **POSITIVE DETECTION:** High risk of cardiovascular disease. Clinical follow-up required.")
            else:
                st.success("✅ **NEGATIVE DETECTION:** Parameters within standard healthy ranges.")
        except Exception as e:
            st.error(f"System Fault: {e}")

with t2:
    priority = {'Random Forest': 5, 'Logistic Regression': 4, 'SVM': 3, 'Decision Tree': 2, 'KNN': 1}
    best_algo = max(results, key=lambda k: (float(results[k]['Accuracy']), float(results[k]['F1-Score']), priority[k]))
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Model", best_algo)
    col2.metric("Accuracy", f"{results[best_algo]['Accuracy']}%")
    col3.metric("F1-Score", f"{results[best_algo]['F1-Score']}%")
    col4.metric("Engine State", "Online")
    
    st.markdown("<br>", unsafe_allow_html=True)
    df_res = pd.DataFrame(results).T.reset_index()
    df_res.columns = ['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    st.dataframe(df_res.style.highlight_max(axis=0, color='#1A4E29'), use_container_width=True)

with t3:
    st.markdown("""
    ### Engineering Specifications
    This system utilizes an auto-healing machine learning pipeline capable of dynamic artifact generation. It features a hierarchical tie-breaking protocol to ensure optimal model selection across multi-dimensional cardiovascular data.
    
    **Development Team:** Arpan Das, Chandan Kumar Mishra, MD Belal  
    **Affiliation:** BACET
    """)
