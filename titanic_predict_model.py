import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Function to load the trained Naive Bayes model
def load_model(model_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Function to predict survival using the loaded model
def predict_survival(model, features):
    # Preprocess the input features
    features = np.array(features).reshape(1, -1)
    # Predict survival
    prediction = model.predict(features)
    return prediction

def main():
    # Load the trained Naive Bayes model
    model_path = 'models/NaiveBayes_model.pkl'  # Path to the saved model
    nb_model = load_model(model_path)

    # Load sample Titanic dataset
    data_path = 'train.csv'  # Path to your sample Titanic dataset
    df = pd.read_csv(data_path)

    # Displaying an animated image at the top
    st.image("https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExazVtZnd0Z3ZzeWdqZGpjNm5iazQ4eG5nYm12Y2QzY2EwYXc5NGdsYyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/OJw4CDbtu0jde/100.webp", use_column_width=True)

    # Giving a title with colorful text and custom font
    st.markdown(
        "<h1 style='text-align: center; color: #ff6347; font-family: Arial, sans-serif;'>Titanic Survival Prediction Web App</h1>", 
        unsafe_allow_html=True
    )

    # Title and description with colorful fonts and custom font
    st.title("🚢 Titanic Survival Prediction 🌊")
    st.markdown(
        """
        This app predicts whether a passenger survived the Titanic disaster based on various features.
        """,
        unsafe_allow_html=True
    )

    # Input features with colorful sidebar and custom font
    st.sidebar.header("📋 Input Features 🖊️")

    st.sidebar.markdown("<h3 style='color: #FFA07A; font-family: Arial, sans-serif;'>Age</h3>", unsafe_allow_html=True)
    age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30, label_visibility='collapsed', help="Enter the age of the passenger.")

    st.sidebar.markdown("<h3 style='color: #20B2AA; font-family: Arial, sans-serif;'>Ticket Class (PClass)</h3>", unsafe_allow_html=True)
    pclass = st.sidebar.selectbox("Ticket Class (PClass)", [1, 2, 3], label_visibility='collapsed', help="Select the ticket class of the passenger.")

    st.sidebar.markdown("<h3 style='color: #9370DB; font-family: Arial, sans-serif;'>Number of Siblings/Spouses Aboard (SibSp)</h3>", unsafe_allow_html=True)
    sibsp = st.sidebar.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0, label_visibility='collapsed', help="Enter the number of siblings or spouses aboard.")

    st.sidebar.markdown("<h3 style='color: #FF6347; font-family: Arial, sans-serif;'>Number of Parents/Children Aboard (Parch)</h3>", unsafe_allow_html=True)
    parch = st.sidebar.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0, label_visibility='collapsed', help="Enter the number of parents or children aboard.")

    st.sidebar.markdown("<h3 style='color: #4682B4; font-family: Arial, sans-serif;'>Fare</h3>", unsafe_allow_html=True)
    fare = st.sidebar.number_input("Fare", min_value=0.0, max_value=1000.0, value=30.0, step=0.01, label_visibility='collapsed', help="Enter the fare paid by the passenger.")

    st.sidebar.markdown("<h3 style='color: #32CD32; font-family: Arial, sans-serif;'>Sex</h3>", unsafe_allow_html=True)
    sex = st.sidebar.selectbox("Sex", ["male", "female"], label_visibility='collapsed', help="Select the gender of the passenger.")

    # Map sex to binary values (0 for female, 1 for male)
    sex = 1 if sex == "male" else 0

    # Prediction button with colorful style and custom font
    if st.sidebar.button("🔮 Predict 🔍"):
        # Collect input features
        input_features = [age, pclass, sibsp, parch, fare, sex]
        # Predict survival
        prediction = predict_survival(nb_model, input_features)
        # Display prediction with colorful font and custom font
        if prediction[0] == 1:
            st.markdown("<h3 style='color: #32CD32; font-family: Arial, sans-serif;'>🎉 Based on the input features, the passenger is predicted to have survived!</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color: #FF6347; font-family: Arial, sans-serif;'>⚠️ Based on the input features, the passenger is predicted to have not survived.</h3>", unsafe_allow_html=True)

    # Comparison section
    st.sidebar.header("🔍 Compare with Actual Data 📊")
    passenger = st.sidebar.selectbox("Select a Passenger", df['Name'].dropna().tolist())
    passenger_data = df[df['Name'] == passenger].iloc[0]

    st.markdown("## Comparison with Actual Passenger Data")
    st.write(f"**Selected Passenger: {passenger_data['Name']}**")
    st.write(f"**Age:** {passenger_data['Age']}")
    st.write(f"**Ticket Class (PClass):** {passenger_data['Pclass']}")
    st.write(f"**Number of Siblings/Spouses Aboard (SibSp):** {passenger_data['SibSp']}")
    st.write(f"**Number of Parents/Children Aboard (Parch):** {passenger_data['Parch']}")
    st.write(f"**Fare:** {passenger_data['Fare']}")
    st.write(f"**Sex:** {'male' if passenger_data['Sex'] == 1 else 'female'}")
    st.write(f"**Survived:** {'Yes' if passenger_data['Survived'] == 1 else 'No'}")

    # Displaying another image at the bottom
    st.image("https://wallpapercave.com/wp/wp3307707.jpg", use_column_width=True)


    # Feedback Mechanism
    st.sidebar.header("💌 Feedback 📝")
    feedback = st.sidebar.text_area("Please share your feedback here:")
    if st.sidebar.button("Submit Feedback"):
        # You can add code here to store the feedback in a database or send it via email
        st.sidebar.success("Thank you for your feedback!")

if __name__ == "__main__":
    main()
