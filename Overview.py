import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

## data

df = pd.read_csv('weather.csv')

df_proc = pd.read_csv('finaldata.csv')

st.set_page_config(page_title="Homepage",layout="wide")
#side bar
st.sidebar.header("PENGOLAHAN DATA LAPORAN CUACA DI SEATTLE DENGAN ALGORITMA KNN")
#st.sidebar.image("1835901.jpg")

##layout

st.title(''' PENGOLAHAN DATA LAPORAN CUACA DI SEATTLE DENGAN ALGORITMA KNN ''')
st.write('Bagus Rahma Aulia Chandra - A11.2017.10295')
st.markdown('''   Dataset yang akan dianalisa adalah laporan cuaca wilayah seattle selama 3 tahun dari 1 januari 2012 hingga 31 desember 2015,dataset laporan cuaca kali ini memiliki fitur  yakni :
- Temperatur minimal
- Temperatur maximal
- Pengendapan (precipation)
- Kecepatan angin
 ''')

st.write(df)

st.markdown(''' Setelah dilakukan proses data cleaning dan data processing antara lain:  
            - Pengecheckan NULL Value
            - Menghapus kolom yang tidak diperlukan
            - Mengidentifikasi Target Class
            - Ditampilkan dalam bentuk diagram (Visualisasi Data
            - Pemrosesan dengan menggunakan algoritma KNN
        
Berikut adalah contoh data yang sudah di Grouping berdasar Kolom Dates dibuang ''')

st.write(df_proc)