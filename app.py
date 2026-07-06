import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/student_model.pkl")

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

st.sidebar.title("Student Performance Prediction")
st.sidebar.write("Machine Learning Internship Project")
st.sidebar.write("Developed by Jahnavi")

st.title("🎓 Student Performance Prediction System")

st.info("Enter the student details below and click Predict Performance.")

hours = st.number_input(
    "Hours Studied",
    min_value=0,
    max_value=12,
    value=5
)

previous_scores = st.number_input(
    "Previous Scores",
    min_value=0,
    max_value=100,
    value=75
)

activities = st.selectbox(
    "Extracurricular Activities",
    ["Yes", "No"]
)

sleep = st.number_input(
    "Sleep Hours",
    min_value=0,
    max_value=12,
    value=7
)

papers = st.number_input(
    "Sample Question Papers Practiced",
    min_value=0,
    max_value=20,
    value=5
)

activity = 1 if activities == "Yes" else 0

if st.button("Predict Performance"):

    input_data = pd.DataFrame({
        "Hours Studied": [hours],
        "Previous Scores": [previous_scores],
        "Extracurricular Activities": [activity],
        "Sleep Hours": [sleep],
        "Sample Question Papers Practiced": [papers]
    })

    prediction = model.predict(input_data)[0]

    st.subheader("Entered Details")
    st.write(input_data)

    st.subheader("Prediction Result")

    st.success(f"Predicted Performance Index: {prediction:.2f}")

    st.progress(min(int(prediction), 100))

    if prediction >= 90:
        st.success("🌟 Excellent Performance")
    elif prediction >= 75:
        st.info("👍 Good Performance")
    elif prediction >= 50:
        st.warning("🙂 Average Performance")
    else:
        st.error("📚 Needs Improvement")

with st.expander("How to use this app"):
    st.write("""
1. Enter Hours Studied.
2. Enter Previous Scores.
3. Select Extracurricular Activities.
4. Enter Sleep Hours.
5. Enter Sample Question Papers Practiced.
6. Click Predict Performance.
""")

st.markdown("---")
st.write("Developed by Jahnavi")