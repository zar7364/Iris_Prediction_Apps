import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Ayo Prediksi Jenis Bunga Iris!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Panjang Mahkota Bunga', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Lebar Mahkota Bunga', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Panjang Kuncup', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Lebar Kuncup', 0.1, 2.5, 0.2)
    data = {'Panjang Mahkota Bunga': sepal_length,
            'Lebar Mahkota Bunga': sepal_width,
            'Panjang Kuncup': petal_length,
            'Lebar Kuncup': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Masukan Datamu!')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Kelas Bunga')
st.write(iris.target_names)

st.subheader('Prediksi')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Peluang Prediksi dalam %')
st.write(prediction_proba)
