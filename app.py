import streamlit as st
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://bharatimudigoudra912000:912000@cluster0.no155.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
# create a database 
db = client['student']
# In mongodb data collection 
collection = db["Iris_data"]


def load_models():
    model_files = {
        "SVC Binary": "svm_binary.pkl",
        "SVC Multi": "svm_multi.pkl",
        "Logistic Regression Binary": "logistics_binary.pkl",
        "Logistic Regression OVR": "logistics_ovr.pkl",
        "Logistic Regression Multinomial": "logistics_multinomial.pkl"
    }
    models = {}

    for model_name, file_name in model_files.items():
        if os.path.exists(file_name):
            with open(file_name, "rb") as file:
                models[model_name] = joblib.load(file)
        else:
            print(f"Warning: {file_name} not found. Skipping {model_name}.")
    return models

def predict(model, data):
    df = pd.DataFrame([data])  # Convert user input to DataFrame
    prediction = model.predict(df)  # Directly predict without scaling
    return prediction

def main():
    st.title("Iris Flower Classification")
    st.write("Enter the features to classify the iris flower.")
    
    models = load_models()
    model_selection = st.sidebar.selectbox("Select Model", list(models.keys()))
    selected_model = models[model_selection]
    
    sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.8)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=5.0, value=3.0)
    petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.0)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.2)
    
    if st.button("Predict Class"):
        user_data = {
            "sepal length (cm)": sepal_length,
            "sepal width (cm)": sepal_width,
            "petal length (cm)": petal_length,
            "petal width (cm)": petal_width
        }
        
        prediction = predict(selected_model, user_data)
        iris_data = load_iris()
        predicted_class = iris_data.target_names[prediction[0]]
        st.success(f"Predicted Iris Class: {predicted_class}")
    
    if st.sidebar.button("Select Model"):
        st.sidebar.success(f"You selected {model_selection}")

if __name__ == "__main__":
    main()
