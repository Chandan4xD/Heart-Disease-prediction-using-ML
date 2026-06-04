import streamlit as st
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="CardioGuard AI")

@st.cache_resource(show_spinner="Training models, please wait...")
def load_or_train():
    if all(os.path.exists(f) for f in ['best_model.pkl','scaler.pkl','results.pkl','feature_names.pkl']):
        model        = pickle.load(open('best_model.pkl','rb'))
        scaler       = pickle.load(open('scaler.pkl','rb'))
        results      = pickle.load(open('results.pkl','rb'))
        feature_names= pickle.load(open('feature_names.pkl','rb'))
        return model, scaler, results, feature_names

    df = pd.read_csv('heart.csv')
    df.columns = df.columns.str.strip()
    if 'patientid' in df.columns:
        df.drop('patientid', axis=1, inplace=True)
    if 'Classification' in df.columns and 'target' not in df.columns:
        df.rename(columns={'Classification': 'target'}, inplace=True)

    X = df.drop('target', axis=1)
    y = df['target']
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        'Random Forest' : RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree' : DecisionTreeClassifier(random_state=42),
        'SVM'           : SVC(kernel='rbf', probability=True, random_state=42),
        'KNN'           : KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes'   : GaussianNB()
    }

    results = {}
    trained = {}
    for name, clf in models.items():
        clf.fit(X_train_s, y_train)
        preds = clf.predict(X_test_s)
        results[name] = {
            'Accuracy' : round(accuracy_score(y_test, preds)  * 100, 2),
            'Precision': round(precision_score(y_test, preds) * 100, 2),
            'Recall'   : round(recall_score(y_test, preds)    * 100, 2),
            'F1-Score' : round(f1_score(y_test, preds)        * 100, 2),
        }
        trained[name] = clf

    best_name = max(results, key=lambda k: (results[k]['Accuracy'], results[k]['F1-Score']))
    return trained[best_name], scaler, results, feature_names

try:
    model, scaler, results, feature_names = load_or_train()
except Exception as e:
    st.error(f"Failed to load or train model: {e}")
    st.stop()

page = st.sidebar.selectbox("Navigation", ["Prediction", "Performance", "About"])

if page == "Prediction":
    st.title("❤️ CardioGuard AI")
    st.subheader("Heart Disease Prediction System")

    age              = st.number_input("Age", 1, 100, 40)
    gender           = st.selectbox("Gender", [0, 1])
    chestpain        = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    restingBP        = st.number_input("Resting BP", 50, 250, 120)
    serumcholestrol  = st.number_input("Serum Cholesterol", 0, 600, 200)
    fastingbloodsugar= st.selectbox("Fasting Blood Sugar", [0, 1])
    restingrelectro  = st.selectbox("Resting ECG", [0, 1, 2])
    maxheartrate     = st.number_input("Max Heart Rate", 50, 250, 150)
    exerciseangia    = st.selectbox("Exercise Angina", [0, 1])
    oldpeak          = st.number_input("Old Peak", 0.0, 10.0, 1.0)
    slope            = st.selectbox("Slope", [0, 1, 2, 3])
    noofmajorvessels = st.selectbox("No. of Major Vessels", [0, 1, 2, 3])

    if st.button("Predict"):
        input_dict = {
            'age': age, 'gender': gender, 'chestpain': chestpain,
            'restingBP': restingBP, 'serumcholestrol': serumcholestrol,
            'fastingbloodsugar': fastingbloodsugar, 'restingrelectro': restingrelectro,
            'maxheartrate': maxheartrate, 'exerciseangia': exerciseangia,
            'oldpeak': oldpeak, 'slope': slope, 'noofmajorvessels': noofmajorvessels
        }
        input_data = np.array([[input_dict[col] for col in feature_names]])
        scaled     = scaler.transform(input_data)
        prediction = model.predict(scaled)[0]

        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

elif page == "Performance":
    st.title("📊 Model Performance")
    df_res = pd.DataFrame(results).T.reset_index()
    df_res.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    st.dataframe(df_res.style.highlight_max(axis=0, color='lightgreen'),
                 use_container_width=True)

else:
    st.title("ℹ️ About Project")
    st.write("CardioGuard AI is a Machine Learning based Heart Disease Prediction System.")
    st.subheader("👨‍💻 Team Members")
    st.write("1. Chandan Kumar Mishra")
    st.write("2. MD Belal")
    st.subheader("🛠️ Technologies Used")
    st.write("Python | Streamlit | Pandas | Scikit-Learn")
