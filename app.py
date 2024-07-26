import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier

model = joblib.load('predicting_anemia.pkl')

class_names = {
    0: 'Healthy',
    1: 'Iron deficiency anemia',
    2: 'Leukemia',
    3: 'Macrocytic anemia',
    4: 'Normocytic hypochromic anemia',
    5: 'Normocytic normochromic anemia',
    6: 'Other microcytic anemia',
    7: 'Thrombocytopenia'
}

def preprocess_input(data):
    df = pd.DataFrame(data, index=[0])
    return df

st.markdown("""
    <div style="background-color:#B19CD9;padding:10px;border-radius:10px">
    <h1 style="color:white;text-align:center;">Anemia Diagnosis Prediction</h1>
    </div>
""", unsafe_allow_html=True)

st.write("""
Did you know that there are many types of anemia? 
""")

st.write("""
If you have anemia, this app will predict the type of anemia that you have.
""")

st.sidebar.header("Patient Data Inputs")
hgb = st.sidebar.number_input("Hemoglobin (HGB)", format="%.1f")
mcv = st.sidebar.number_input("Mean Corpuscular Volume (MCV)", format="%.1f")
mch = st.sidebar.number_input("Mean Corpuscular Hemoglobin (MCH)", format="%.1f")
mchc = st.sidebar.number_input("Mean Corpuscular Hemoglobin Concentration (MCHC)", format="%.1f")
plt = st.sidebar.number_input("Platelet Count (PLT)", format="%.1f")
rbc = st.sidebar.number_input("Red Blood Cell Count (RBC)", format="%.1f")
wbc = st.sidebar.number_input("White Blood Cell Count (WBC)", format="%.1f")

def user_input_features():
    data = {
        'HGB': hgb,
        'MCV': mcv,
        'MCH': mch,
        'MCHC': mchc,
        'PLT': plt,
        'RBC': rbc,
        'WBC': wbc
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

preprocessed_data = preprocess_input(df)

if st.sidebar.button("Predict"):
    prediction = model.predict(preprocessed_data)
    prediction_proba = model.predict_proba(preprocessed_data)
    
    predicted_class = class_names[prediction[0]]
    
    st.subheader('Prediction')
    st.write(f"Prediction: {predicted_class}")
    
    st.subheader('Prediction Probability')
    st.write(prediction_proba)
