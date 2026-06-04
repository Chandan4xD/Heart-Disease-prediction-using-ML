import streamlit as st
import pickle
import pandas as pd

# Load trained model
model = pickle.load(open("heart_model.pkl", "rb"))

st.set_page_config(page_title="CardioGuard AI")

# Sidebar
page = st.sidebar.selectbox(
    "Navigation",
    ["Prediction", "About"]
)

# Prediction Page
if page == "Prediction":

    st.title("❤️ CardioGuard AI")
    st.subheader("Heart Disease Prediction System")

    age = st.number_input("Age", 1, 100, 40)
    gender = st.selectbox("Gender", [0, 1])

    chestpain = st.selectbox("Chest Pain Type", [0, 1, 2, 3])

    restingBP = st.number_input("Resting BP", 50, 250, 120)

    serumcholestrol = st.number_input("Serum Cholesterol", 0, 600, 200)

    fastingbloodsugar = st.selectbox("Fasting Blood Sugar", [0, 1])

    restingrelectro = st.selectbox("Resting ECG", [0, 1, 2])

    maxheartrate = st.number_input("Max Heart Rate", 50, 250, 150)

    exerciseangia = st.selectbox("Exercise Angina", [0, 1])

    oldpeak = st.number_input("Old Peak", 0.0, 10.0, 1.0)

    slope = st.selectbox("Slope", [0, 1, 2, 3])

    noofmajorvessels = st.selectbox("No. of Major Vessels", [0, 1, 2, 3])

    if st.button("Predict"):

        input_data = pd.DataFrame([[
            age,
            gender,
            chestpain,
            restingBP,
            serumcholestrol,
            fastingbloodsugar,
            restingrelectro,
            maxheartrate,
            exerciseangia,
            oldpeak,
            slope,
            noofmajorvessels
        ]])

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

# About Page
else:

    st.title("ℹ️ About Project")

    st.write("""
    CardioGuard AI is a Machine Learning based Heart Disease Prediction System.

    Features:
    - Heart Disease Prediction
    - Machine Learning Model
    - User Friendly Interface
    - Streamlit Web Application
    """)

    st.subheader("👨‍💻 Team Members")

    st.write("1. Arpan Das")
    st.write("2. Team Member 2")
    st.write("3. Team Member 3")

    st.subheader("🛠️ Technologies Used")

    st.write("- Python")
    st.write("- Streamlit")
    st.write("- Pandas")
    st.write("- Scikit-Learn")
    st.write("- Machine Learning")
