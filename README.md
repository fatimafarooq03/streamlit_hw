## Insurance Cost Prediction App
This application uses a machine learning model to predict insurance costs based on user input. It leverages a trained model to provide cost estimates based on features such as age, sex, BMI, number of children, smoking status, and region.

### Key Features:
User Input: Enter details such as age, sex, BMI, number of children, smoking status, and region.
Prediction: The app uses a pre-trained model to estimate insurance costs based on the input data.
Interactive Interface: Built with Streamlit for a user-friendly interface that displays predictions in real-time.
### Installation and Setup:
Clone the repository.
Install the required dependencies using pip install -r requirements.txt.
Place your pre-trained model (insurance_model.pkl) in the root directory.
Run the app with streamlit run app.py.
### Dependencies:
streamlit
pandas
scikit-learn
Other libraries specified in requirements.txt
### Notes:
Ensure the model file (insurance_model.pkl) matches the expected format and dependencies.
Update the library versions if you encounter compatibility issues.
