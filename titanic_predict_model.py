import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

# Function to load the trained Naive Bayes model
def load_model(model_path):
    try:
        if not os.path.exists(model_path):
            st.error(f"The model file was not found at path: {model_path}")
            return None
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

model = load_model('NaiveBayes_model.pkl')

if model is None:
    st.stop()

st.title("Titanic Survival Prediction")

st.sidebar.header("Passenger Details")
age = st.sidebar.slider("Age", 0, 80, 29)
pclass = st.sidebar.selectbox("Class", [1, 2, 3])
sibsp = st.sidebar.selectbox("Siblings/Spouses Aboard", [0, 1, 2, 3, 4, 5, 8])
parch = st.sidebar.selectbox("Parents/Children Aboard", [0, 1, 2, 3, 4, 5, 6])
fare = st.sidebar.slider("Fare", 0, 550, 32)
sex = st.sidebar.radio("Gender", ('male', 'female'))
embarked = st.sidebar.radio("Port of Embarkation", ('C', 'Q', 'S'))

if st.sidebar.button("Predict"):
    data = pd.DataFrame({
        'Pclass': [pclass],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Sex_female': [1 if sex == 'female' else 0],
        'Sex_male': [1 if sex == 'male' else 0],
        'Embarked_C': [1 if embarked == 'C' else 0],
        'Embarked_Q': [1 if embarked == 'Q' else 0],
        'Embarked_S': [1 if embarked == 'S' else 0]
    })

    try:
        prediction = model.predict(data)
        prediction_proba = model.predict_proba(data)

        st.write(f"Predicted Survival: {'Yes' if prediction[0] == 1 else 'No'}")
        st.write(f"Probability of Survival: {prediction_proba[0][1]:.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
