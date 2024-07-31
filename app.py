import streamlit as st
import numpy as np
import pickle
import pandas as pd

st.header('A Model for Insurance Costs')

st.write('Please enter the Age, Sex, BMI, number of children, smoking status, and region')

age = st.number_input('Age')
sex = st.selectbox('Sex', ['male', 'female'])
BMI = st.number_input('BMI')
children = st.number_input('Number of Children')
smoker = st.selectbox('Do you smoke?', ['yes', 'no'])
region = st.selectbox('Region', ['southwest', 'northwest','southeast','northeast'])

# Create feature vector as a DataFrame
X = pd.DataFrame([[age, sex, BMI, children, smoker, region]], columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

# Load the trained model
with open('insurance_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make a prediction
pred = model.predict(X)

# Display the prediction
st.write(f'The model predicts an insurance cost of $ {pred[0]}')
