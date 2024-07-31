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

# Perform one-hot encoding for categorical variables
X_encoded = pd.get_dummies(X, columns=['sex', 'smoker', 'region'])

# Ensure the encoded dataframe has all the expected columns in the correct order
expected_columns = ['age', 'bmi', 'children', 'sex_female', 'sex_male', 'smoker_no', 'smoker_yes',
                    'region_northwest', 'region_southeast', 'region_southwest', 'region_northeast']
X_encoded = X_encoded.reindex(columns=expected_columns, fill_value=0)

# Load the trained model
with open('insurance_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make a prediction
pred = model.predict(X_encoded)

# Display the prediction
st.write(f'The model predicts an insurance cost of $ {pred[0]}')
