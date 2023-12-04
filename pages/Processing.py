import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('finaldata.csv')

st.title(''' PENGOLAHAN DATA LAPORAN CUACA DI SEATTLE DENGAN ALGORITMA KNN ''')
st.write('Prediksi Data Baru')

input_precipation = st.number_input ("Precipation")
#min_value=df('precipation').min()
#max_value=df('precipation').max()

input_min_temp = st.number_input ("Minimum Temperature")
#min_value=df('precipation').min()
#max_value=df('precipation').max()

input_max_temp = st.number_input ("Maximum Temperature")
#min_value=df('precipation').min()
#max_value=df('precipation').max()

input_wind = st.number_input ("Wind Velocity")
#min_value=df('precipation').min()
#max_value=df('precipation').max()

result = "-"

X = df.iloc[:, :-1]
y = df['weather']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

Predict = st.button("Predict")
        
if Predict:
    if input_precipation != str(0.00) and input_min_temp != str(0.00) and input_max_temp != str(0.00) and input_wind != str(0.00):
        precipation = float(input_precipation)
        min_temp = float(input_min_temp)
        max_temp = float(input_max_temp)
        wind = float(input_wind)
        prediction = knn.predict([[precipation,min_temp,max_temp,wind]])[0]
        result = str(prediction)
        if result =='0':
            st.subheader(f"Prediction : {result}")
            st.write('Cuaca sedang Gerimis')
        elif result =='1':
            st.subheader(f"Prediction : {result}")
            st.write('Cuaca sedang Hujan')
        elif result =='2':
            st.subheader(f"Prediction : {result}")
            st.write('Cuaca sedang Cerah')
        elif result =='3':
            st.subheader(f"Prediction : {result}")
            st.write('Cuaca sedang Bersalju')
        elif result =='4':
            st.subheader(f"Prediction : {result}")
            st.write('Cuaca sedang Berkabut')
    else:
        result = "-"
else :
    result = "Please complete form above!"
