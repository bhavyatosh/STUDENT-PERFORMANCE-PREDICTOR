# app.py

import streamlit as st
import numpy as np
from model import load_model_and_scaler

# Load model and scaler from Google Drive
model, scaler = load_model_and_scaler()

st.title("🎓 Student Performance Predictor")

# Input UI
gender = st.selectbox("Gender", ["female", "male"])
race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parent_edu = st.selectbox("Parental Level of Education", [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
prep_course = st.selectbox("Test Preparation Course", ["none", "completed"])
math_score = st.slider("Math Score", 0, 100, 70)
reading_score = st.slider("Reading Score", 0, 100, 70)
writing_score = st.slider("Writing Score", 0, 100, 70)

def encode_input():
    cat_dict = {
        "gender": {"female": 0, "male": 1},
        "race": {"group A": 0, "group B": 1, "group C": 2, "group D": 3, "group E": 4},
        "parent_edu": {
            "some high school": 0, "high school": 1, "some college": 2,
            "associate's degree": 3, "bachelor's degree": 4, "master's degree": 5
        },
        "lunch": {"free/reduced": 0, "standard": 1},
        "prep": {"none": 0, "completed": 1}
    }
    return [
        cat_dict["gender"][gender],
        cat_dict["race"][race],
        cat_dict["parent_edu"][parent_edu],
        cat_dict["lunch"][lunch],
        cat_dict["prep"][prep_course],
        math_score,
        reading_score,
        writing_score
    ]

if st.button("Predict Performance"):
    features = np.array([encode_input()])
    scaled_input = scaler.transform(features)
    prediction = model.predict(scaled_input)
    st.success(f"🎯 Predicted Performance: **{prediction[0]}**")
