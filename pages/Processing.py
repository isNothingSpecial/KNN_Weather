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

st.number_input ("Precipation")
#min_value=df('precipation').min()
#max_value=df('precipation').max()

st.number_input ("Minimum Temperature")
#min_value=df('precipation').min()
#max_value=df('precipation').max()

st.number_input ("Maximum Temperature")
#min_value=df('precipation').min()
#max_value=df('precipation').max()

st.number_input ("Wind Velocity")
#min_value=df('precipation').min()
#max_value=df('precipation').max()

st.button("Predict")

X = df.iloc[:, :-1]
y = df['weather']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


st.subheader("Prediction : ")
