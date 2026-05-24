import os
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
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
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
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

st.title("Heart Disease Prediction System")
st.caption("Developed by Arpan, Chandan & MD Belal | Indian Cardiovascular Dataset")

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
    
    if st.button("Predict Result", type="primary"):
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
            
            if pred == 1: st.error("⚠️ Heart Disease Detected")
            else: st.success("✅ No Heart Disease Detected")
        except KeyError as e:
            st.error(f"Dataset column mismatch. Could not find column: {e}. Please ensure you are using the correct Mendeley dataset.")

with tab2:
    st.write("### Model Performance Metrics")
    df_res = pd.DataFrame(results).T.reset_index()
    df_res.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
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
    * **Inference Engine:** An automated pipeline evaluating multiple classification algorithms (Random Forest, Decision Tree, Naive Bayes, KNN, Logistic Regression) to deploy the optimal predictive model.
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
