import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.write("""
# Credit Card Defaulter Prediction
This app predicts if an Credit Card user is going to stop in the next month
""")

st.sidebar.header('User Input Parameters')

uploaded_file = st.sidebar.file_uploader("Upload your file here:", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input():
        Education1 = st.sidebar.slider('Education',1,4,1)
        Marriage1 = st.sidebar.slider('Marriage',1 ,2 ,1)
        Age1 = st.sidebar.slider('Age', 10, 60, 10)
        LIMIT_BAL1 = st.sidebar.slider('LIMIT_BAL', 100, 1000000, 100)
        PAY_11 = st.sidebar.slider('PAY_1', 0, 100000, 0)
        BILL_AMT_11 = st.sidebar.slider('BILL_AMT_1', -165580, 964511,-165580 )
        BILL_AMT_21 = st.sidebar.slider('BILL_AMT_2', -69777, 983931, -69777)
        BILL_AMT_31 = st.sidebar.slider('BILL_AMT_3', -157264, 1664089, -157264)
        data = {
        'Education' : Education1,
        'Marriage' : Marriage1,
        'Age' : Age1,
        'LIMIT_BAL' : LIMIT_BAL1,
        'PAY_1' : PAY_11,
        'BILL_AMT_1' : BILL_AMT_11,
        'BILL_AMT_2' : BILL_AMT_21,
        'BILL_AMT_3' : BILL_AMT_31,
        }
        features = pd.DataFrame(data, index=[0])
        return features               
    df = user_input()

dataf = pd.read_csv('cleaned_data.csv')
features_response = dataf.columns.tolist()
items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 'none',
                   'others', 'university','BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                    'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
features_response = [item for item in features_response if item not in items_to_remove]

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below)')
    st.write(df)

X_train, X_test, y_train, y_test = \
train_test_split(dataf[features_response[:-1]].values, dataf['default payment next month'].values,
test_size=0.2, random_state=24)

rf = RandomForestClassifier\
(n_estimators=200, criterion='gini', max_depth=9,
min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None,
random_state=4, verbose=1, warm_start=False, class_weight=None)

rf.fit(X_train, y_train)
pred = rf.predict(df)
pred_proba =rf.predict_proba(df)

st.write("Accuracy")
st.write(pred)

st.write(y_test)
