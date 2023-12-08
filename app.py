import streamlit as st
import numpy as np
import pickle
import pandas as pd

from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay 
from sklearn.compose import make_column_transformer 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

st.header('A Model for insurance costs')

st.write('Please enter the Age, Sex, BMI, number of children, smoking status, and region')

age = st.number_input('Age')
sex = st.selectbox('Sex', ['male', 'female'])  # Use a dropdown for categorical variable
BMI = st.number_input('BMI')
children = st.number_input('Number of Children')
smoker = st.selectbox('Do you smoke?', ['yes', 'no'])  # Use a dropdown for categorical variable
region = st.selectbox('Region', ['southwest', 'northwest','southeast','northeast'])


# Create feature vector
X = np.array([[age, sex, BMI, children, smoker,region]])
X = pd.DataFrame(X,columns=['age','sex','bmi','children','smoker','region'])



with open('insurance_model.pkl', 'rb') as f:
    model = pickle.load(f)

pred = model.predict(X)

st.write(f'The model predicts an insurance cost of $ {pred[0]}')
 
