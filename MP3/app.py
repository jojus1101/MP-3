import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models
lin_reg = joblib.load('Models/linear_regression_income.pkl')
clf = joblib.load('Models/attrition_classifier.pkl')
kmeans = joblib.load('Models/kmeans_clustering.pkl')

# Columns and categorical encoders used during training (adjust as needed)
cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

st.title("Employee Behavior Prediction")

model_choice = st.selectbox("Select Model", 
                            ["Linear Regression: Predict Monthly Income", 
                             "Classification: Predict Attrition", 
                             "Clustering: Employee Segmentation"])

def encode_categorical_inputs(inputs):
    # Dummy example of encoding (replace with actual LabelEncoder mappings from training)
    encodings = {
        'BusinessTravel': {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2},
        'Department': {'Sales': 0, 'Research & Development': 1, 'Human Resources': 2},
        'EducationField': {'Life Sciences': 0, 'Other': 1, 'Medical': 2, 'Marketing': 3, 'Technical Degree': 4, 'Human Resources': 5},
        'Gender': {'Male': 1, 'Female': 0},
        'JobRole': {
            'Sales Executive':0, 'Research Scientist':1, 'Laboratory Technician':2, 'Manufacturing Director':3,
            'Healthcare Representative':4, 'Manager':5, 'Sales Representative':6, 'Research Director':7, 'Human Resources':8
        },
        'MaritalStatus': {'Single': 0, 'Married': 1, 'Divorced': 2},
        'OverTime': {'Yes': 1, 'No': 0}
    }
    for cat_col in cat_cols:
        if cat_col in inputs:
            inputs[cat_col] = encodings.get(cat_col, {}).get(inputs[cat_col], 0)
    return inputs

if model_choice == "Linear Regression: Predict Monthly Income":
    st.header("Predict Monthly Income")
    input_data = {}
    input_data['Age'] = st.number_input("Age", min_value=18, max_value=65, value=30)
    input_data['BusinessTravel'] = st.selectbox("Business Travel", ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
    input_data['Department'] = st.selectbox("Department", ['Sales', 'Research & Development', 'Human Resources'])
    input_data['DistanceFromHome'] = st.number_input("Distance From Home (miles)", min_value=0, max_value=100, value=10)
    input_data['Education'] = st.slider("Education Level (1-5)", 1, 5, 3)
    input_data['Gender'] = st.selectbox("Gender", ['Male', 'Female'])
    input_data['JobRole'] = st.selectbox("Job Role", [
        'Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
        'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
    input_data['MaritalStatus'] = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
    input_data['NumCompaniesWorked'] = st.number_input("Number of Companies Worked", 0, 10, 2)
    input_data['OverTime'] = st.selectbox("OverTime", ['Yes', 'No'])
    input_data['PercentSalaryHike'] = st.slider("Percent Salary Hike", 0, 50, 10)
    input_data['TotalWorkingYears'] = st.number_input("Total Working Years", 0, 40, 5)
    input_data['TrainingTimesLastYear'] = st.number_input("Training Times Last Year", 0, 10, 2)
    input_data['WorkLifeBalance'] = st.slider("Work Life Balance (1-4)", 1, 4, 3)
    input_data['YearsAtCompany'] = st.number_input("Years at Company", 0, 40, 3)

    input_data_enc = encode_categorical_inputs(input_data)
    input_df = pd.DataFrame([input_data_enc])
    
    if st.button("Predict Monthly Income"):
        pred_income = lin_reg.predict(input_df)[0]
        st.success(f"Predicted Monthly Income: ${pred_income:,.2f}")

elif model_choice == "Classification: Predict Attrition":
    st.header("Predict Employee Attrition")
    input_data = {}
    input_data['Age'] = st.number_input("Age", min_value=18, max_value=65, value=30)
    input_data['BusinessTravel'] = st.selectbox("Business Travel", ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
    input_data['Department'] = st.selectbox("Department", ['Sales', 'Research & Development', 'Human Resources'])
    input_data['DistanceFromHome'] = st.number_input("Distance From Home (miles)", min_value=0, max_value=100, value=10)
    input_data['Education'] = st.slider("Education Level (1-5)", 1, 5, 3)
    input_data['Gender'] = st.selectbox("Gender", ['Male', 'Female'])
    input_data['JobRole'] = st.selectbox("Job Role", [
        'Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
        'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
    input_data['MaritalStatus'] = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
    input_data['NumCompaniesWorked'] = st.number_input("Number of Companies Worked", 0, 10, 2)
    input_data['OverTime'] = st.selectbox("OverTime", ['Yes', 'No'])
    input_data['PercentSalaryHike'] = st.slider("Percent Salary Hike", 0, 50, 10)
    input_data['TotalWorkingYears'] = st.number_input("Total Working Years", 0, 40, 5)
    input_data['TrainingTimesLastYear'] = st.number_input("Training Times Last Year", 0, 10, 2)
    input_data['WorkLifeBalance'] = st.slider("Work Life Balance (1-4)", 1, 4, 3)
    input_data['YearsAtCompany'] = st.number_input("Years at Company", 0, 40, 3)
    
    input_data_enc = encode_categorical_inputs(input_data)
    input_df = pd.DataFrame([input_data_enc])

    if st.button("Predict Attrition"):
        pred_attrition = clf.predict(input_df)[0]
        pred_prob = clf.predict_proba(input_df)[0][1]
        attrition_str = "Yes" if pred_attrition == 1 else "No"
        st.success(f"Predicted Attrition: {attrition_str} (Probability: {pred_prob:.2f})")

else:
    st.header("Employee Segmentation with Clustering")
    st.write("This clusters employees based on selected numeric features.")
    
    cluster_input = {}
    cluster_input['Age'] = st.number_input("Age", min_value=18, max_value=65, value=30)
    cluster_input['DistanceFromHome'] = st.number_input("Distance From Home (miles)", min_value=0, max_value=100, value=10)
    cluster_input['NumCompaniesWorked'] = st.number_input("Number of Companies Worked", 0, 10, 2)
    cluster_input['PercentSalaryHike'] = st.slider("Percent Salary Hike", 0, 50, 10)
    cluster_input['TotalWorkingYears'] = st.number_input("Total Working Years", 0, 40, 5)
    cluster_input['YearsAtCompany'] = st.number_input("Years at Company", 0, 40, 3)

    input_df = pd.DataFrame([cluster_input])
    
    if st.button("Find Cluster"):
        cluster_label = kmeans.predict(input_df)[0]
        st.success(f"Employee belongs to Cluster #{cluster_label}")

