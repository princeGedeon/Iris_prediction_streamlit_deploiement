import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Application pour la prévision des fleurs d'Iris
Cette application prédit la catégorie des fleurs d'Iris

""")

st.sidebar.header("Les parametres d'entrées")

def user_input():
    sepal_length=st.sidebar.slider('La longueur du Sépal',4.3,7.9,5.3)
    sepal_width =st.sidebar.slider('La largeur du Sépal',2.0,7.9,3.3)
    petal_length =st.sidebar.slider('La longueur du Pétal',1.0,7.9,5.3)
    petal_width =st.sidebar.slider('La largeur du Pétal',0.1,7.9,5.3)
    data={
        'sepal_length':sepal_length,
        'sepal_width':sepal_width,
        'petal_length':petal_length,
        'petal_width':petal_width
        }
    fleur_parametres=pd.DataFrame(data,index=[0])
    return fleur_parametres

df=user_input()

st.subheader('On veut trouver la categorie de cette fleur')
st.write(df)
iris=datasets.load_iris()
clf=RandomForestClassifier()
clf.fit(iris.data,iris.target)
prediction=clf.predict(df)

st.subheader("La catégorie de la fleur d'Iris est :")
st.write(iris.target_names[prediction])