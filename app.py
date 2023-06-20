

# Input data files are available in the read-only "../input/" directory

"""
@author: Victor Ajayi
"""
import numpy as np # linear algebra
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from datetime import datetime

import pickle
import streamlit as st

# loading the saved models
model = pickle.load(open('loan_pred.pkl', 'rb'))

# page title
st.title('Loan Prediction using ML')

#Image
st.image('loan.webp')
st.divider() 
# getting the input data from the user
col1, col2, col3, col4, col5 = st.columns(5)
colu1, colu2, colu3, colu4, colu5, colu6 = st.columns(6)
colum1, colum2, colum3, colum4, colum5, colum6 = st.columns(6)
column1, column2, column3, column4, column5, column6 = st.columns(6)
with col1:
    LIMIT_BAL = st.number_input('**:blue[LIMIT BALANCE]**', value=0)
with col2:
    SEX = st.selectbox('**:blue[SEX]**', ['Male', 'Female'])
with col3:
    DOB = st.date_input('**:blue[DOB]**')
with col4:
    EDUCATION = st.selectbox('**:blue[EDUCATION]**', ['Uneducated', 'Undergraduate', 'Graduate','Masters', 'Doctorate'])
with col5:
    MARRIAGE = st.selectbox('**:blue[MARRIAGE]**', ['Single','Married','Divorce'])

with colu1:
    PAY_1 = st.number_input('**:blue[PAY 1]**', value=0, min_value=-2, max_value=8, step=1)
with colu2:
    PAY_2 = st.number_input('**:blue[PAY 2]**', value=0, min_value=-2, max_value=8, step=1)
with colu3:
    PAY_3 = st.number_input('**:blue[PAY 3]**', value=0, min_value=-2, max_value=8, step=1)
with colu4:
    PAY_4 = st.number_input('**:blue[PAY 4]**', value=0, min_value=-2, max_value=8, step=1)
with colu5:
    PAY_5 = st.number_input('**:blue[PAY 5]**', value=0, min_value=-2, max_value=8, step=1)
with colu6:
    PAY_6 = st.number_input('**:blue[PAY 6]**', value=0, min_value=-2, max_value=8, step=1)

with colum1:
    BILL_AMT1 = st.number_input('**:blue[BILL AMOUNT 1]**', value=0)
with colum2:
    BILL_AMT2 = st.number_input('**:blue[BILL AMOUNT 2]**', value=0)
with colum3:
    BILL_AMT3 = st.number_input('**:blue[BILL AMOUNT 3]**', value=0)
with colum4:
    BILL_AMT4 = st.number_input('**:blue[BILL AMOUNT 4]**', value=0)
with colum5:
    BILL_AMT5 = st.number_input('**:blue[BILL AMOUNT 5]**', value=0)
with colum6:
    BILL_AMT6 = st.number_input('**:blue[BILL AMOUNT 6]**', value=0)

with column1:
    PAY_AMT1 = st.number_input('**:blue[PAY AMOUNT 1]**', value=0)
with column2:
    PAY_AMT2 = st.number_input('**:blue[PAY AMOUNT 2]**', value=0)
with column3:
    PAY_AMT3 = st.number_input('**:blue[PAY AMOUNT 3]**', value=0)
with column4:
    PAY_AMT4 = st.number_input('**:blue[PAY AMOUNT 4]**', value=0)
with column5:
    PAY_AMT5 = st.number_input('**:blue[PAY AMOUNT 5]**', value=0)
with column6:
    PAY_AMT6 = st.number_input('**:blue[PAY AMOUNT 6]**', value=0)

#Data Preprocessing
    
data = {
        'LIMIT_BAL': LIMIT_BAL,
        'SEX' : SEX,
        'DOB' : DOB,
        'EDUCATION' : EDUCATION,
        'MARRIAGE': MARRIAGE,
        'PAY_1': PAY_1,
        'PAY_2': PAY_2,
        'PAY_3': PAY_3,
        'PAY_4': PAY_4,
        'PAY_5': PAY_5,
        'PAY_6': PAY_6,
        'PAY_AMT1': PAY_AMT1,
        'PAY_AMT2': PAY_AMT2,
        'PAY_AMT3': PAY_AMT3,
        'PAY_AMT4': PAY_AMT4,
        'PAY_AMT5': PAY_AMT5,
        'PAY_AMT6': PAY_AMT6,
        'BILL_AMT1': BILL_AMT1,
        'BILL_AMT2': BILL_AMT2,
        'BILL_AMT3': BILL_AMT3,
        'BILL_AMT4': BILL_AMT4,
        'BILL_AMT5': BILL_AMT5,
        'BILL_AMT6': BILL_AMT6,  
            }

# oe = OrdinalEncoder(categories = [['Small','Medium','High']])
scaler = StandardScaler()
encoder = LabelEncoder()

def make_prediction(data):
    df = pd.DataFrame(data, index=[0])

    df["LIMIT_BAL"] = np.log(1 + df["LIMIT_BAL"])
    df.SEX = encoder.fit_transform(df.SEX)

    parsed_year = pd.to_datetime(df['DOB'], format='%Y/%m/%d')
    current_year = datetime.now().year
    df['age'] = current_year - parsed_year.dt.year

    edu_order = [['Uneducated', 'Undergraduate', 'Graduate','Masters', 'Doctorate']]
    oe = OrdinalEncoder(categories = edu_order)
    df['EDUCATION'] = oe.fit_transform(df[['EDUCATION']])

    marr_order = [['Single','Married','Divorce']]
    me = OrdinalEncoder(categories = marr_order)
    df['MARRIAGE'] = me.fit_transform(df[['MARRIAGE']])

    df["avg_pay"] = (df["PAY_1"]+df["PAY_2"]+df["PAY_3"]+df["PAY_4"]+df["PAY_5"]+df["PAY_6"])/6
    df['bill_amt'] = (df['BILL_AMT1'] + df['BILL_AMT2'] + df['BILL_AMT3'] + df['BILL_AMT4'] + df['BILL_AMT5'] + df['BILL_AMT6'])
    df['PAY_AMT'] = (df['PAY_AMT1'] + df['PAY_AMT2'] +df['PAY_AMT3'] +df['PAY_AMT4'] +df['PAY_AMT5'] +df['PAY_AMT6'])

    drop_cols = ['DOB','PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6', 'PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
    df = df.drop(drop_cols, axis=1)
    prediction = model.predict(df)
    return prediction
    
final_output = ""
default_out = {0: "no default", 1:"default"}
if st.button('**Predict Loan**'):
    output_prediction = make_prediction(data)
    with st.spinner('Predicting Loan...'):
        time.sleep(1)
        if output_prediction == 0:
            st.success('The Prediction indicates that customer will not default', icon="✅")
        else:
            st.success('The Prediction indicates that customer will default', icon="❌")
st.divider()