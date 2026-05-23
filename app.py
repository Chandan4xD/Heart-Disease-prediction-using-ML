import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

@st.cache_resource
def train_models():
    df = pd.read_csv("heart.csv")
    df.drop(columns=["patientid"], inplace=True, errors="ignore")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=4, random_state=42),
        "SVM": SVC(kernel="rbf", C=2.0, probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Naive Bayes": GaussianNB()
    }

    results = {}
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "Accuracy": round(accuracy_score(y_test, preds) * 100, 2),
            "Precision": round(precision_score(y_test, preds) * 100, 2),
            "Recall": round(recall_score(y_test, preds) * 100, 2),
            "F1-Score": round(f1_score(y_test, preds) * 100, 2)
        }
        trained[name] = model

    best = max(results, key=lambda k: results[k]["Accuracy"])
    return trained[best], scaler, results, best, X_test, y_test


model, scaler, results, best_model_name, X_test, y_test = train_models()

st.title("Heart Disease Prediction System")
st.caption("Dept. of CSE | BACET | Final Year Project")
st.caption("Developed by Arpan Das, Chandan Kumar Mishra & MD Belal")

st.markdown("---")

tab1, tab2 = st.tabs(["Predict", "Model Performance"])

with tab1:
    st.subheader("Enter Patient Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 20, 80, 45)
        gender = st.radio("Gender", ["Female", "Male"], index=1, horizontal=True)
        gender_val = 1 if gender == "Male" else 0
        chestpain = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        restingrelectro = st.selectbox("Resting ECG", [0, 1, 2])

    with col2:
        restingBP = st.number_input("Resting Blood Pressure", 90, 200, 140)
        serumcholestrol = st.number_input("Serum Cholesterol", 0, 610, 250)
        fastingbloodsugar = st.radio("Fasting Blood Sugar > 120", [0, 1], horizontal=True)
        maxheartrate = st.number_input("Max Heart Rate", 60, 220, 150)

    with col3:
        exerciseangia = st.radio("Exercise Induced Angina", [0, 1], horizontal=True)
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.5, 2.0, step=0.1)
        slope = st.selectbox("ST Slope", [0, 1, 2, 3])
        noofmajorvessels = st.selectbox("Major Vessels Coloured", [0, 1, 2, 3])

    st.markdown("")

    if st.button("Predict", type="primary"):
        input_data = pd.DataFrame([{
            "age": age,
            "gender": gender_val,
            "chestpain": chestpain,
            "restingBP": restingBP,
            "serumcholestrol": serumcholestrol,
            "fastingbloodsugar": fastingbloodsugar,
            "restingrelectro": restingrelectro,
            "maxheartrate": maxheartrate,
            "exerciseangia": exerciseangia,
            "oldpeak": oldpeak,
            "slope": slope,
            "noofmajorvessels": noofmajorvessels
        }])

        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0]

        st.markdown("---")
        if prediction == 1:
            st.error(f"Heart Disease Detected — Confidence: {probability[1]*100:.1f}%")
        else:
            st.success(f"No Heart Disease Detected — Confidence: {probability[0]*100:.1f}%")

        st.caption(f"Predicted using {best_model_name} | Accuracy: {results[best_model_name]['Accuracy']}%")
        st.warning("Note: This is a research tool, not a medical diagnosis. Consult a doctor.")

with tab2:
    st.subheader("Model Comparison")

    df_results = pd.DataFrame(results).T.reset_index()
    df_results.columns = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    st.dataframe(df_results, use_container_width=True, hide_index=True)

    st.markdown(f"**Best Model: {best_model_name}** with {results[best_model_name]['Accuracy']}% accuracy")

    st.subheader("Accuracy Comparison")
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["green" if m == best_model_name else "steelblue" for m in df_results["Model"]]
    ax.barh(df_results["Model"], df_results["Accuracy"], color=colors, edgecolor="white")
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 110)
    for i, val in enumerate(df_results["Accuracy"]):
        ax.text(val + 0.5, i, f"{val}%", va="center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.subheader(f"Confusion Matrix — {best_model_name}")
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)

    fig2, ax2 = plt.subplots(figsize=(4, 3))
    im = ax2.imshow(cm, cmap="Blues")
    ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Predicted Healthy", "Predicted Disease"])
    ax2.set_yticklabels(["Actual Healthy", "Actual Disease"])
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=14, fontweight="bold",
                     color="white" if cm[i][j] > cm.max() / 2 else "black")
    fig2.tight_layout()
    st.pyplot(fig2)
