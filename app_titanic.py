import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

URL = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

df = pd.read_csv(URL)

df["sex"] = df["Sex"].replace({"male":0, "female":1})

df = df.select_dtypes(exclude="O")

df.drop(columns=["PassengerId", "Parch"], inplace=True)

esc = MinMaxScaler()

escalated = esc.fit_transform(df[["Age", "Fare"]])

df[["Age", "Fare"]] = escalated

df.dropna(inplace=True)

X = df.drop(columns="Survived")
y = df["Survived"]

lr = LogisticRegression()
lr.fit(X,y)

st.markdown(
    """
    <style>
    .reportview-container {
        background: url(""C:\\Users\\camil\\Downloads\\the-white-star-line-passenger-liner-r-m-s-titanic-embarking-news-photo-1608252641_.jpg"") 
        no-repeat center center fixed;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Titanic')

st.write("""En esta aplicación vas a poder explorar cual sería tu probabilidad de sobrevivir en el Titanic!""")


sex_input = st.text_input("Ingrese su sexo: hombre o mujer")

if sex_input.lower()== "hombre" or sex_input.lower() == "mujer":
    st.write("Has ingresado:", sex_input)
else:
    st.write("Por favor, ingresa tu sexo.")


age_input = st.text_input("Ingrese su edad:")

if age_input:
    st.write("Has ingresado:", age_input)
else:
    st.write("Por favor, ingresa tu edad.")


fare_input = st.text_input("Ingrese el precio de su ticket:")

if fare_input:
    st.write("Has ingresado:", fare_input)
else:
    st.write("Por favor, ingresa el precio de tu ticket.")



sibs_input = st.text_input("Ingrese la cantidad de familiares con los que viaja:")

if sibs_input:
    st.write("Has ingresado:", sibs_input)
else:
    st.write("Por favor, ingresa la cantidad de familiares con los que viaja.")


if sex_input.lower() == "hombre":
    sex_input = 0
elif sex_input.lower() == "mujer":
    sex_input = 1


class_input = st.text_input("Ingrese la clase: 1, 2 o 3")


if class_input.strip():
    try:
        pclass = int(class_input)
    except ValueError:
        st.error("Por favor, ingrese un valor numérico válido para la clase.")
        st.stop()

else:
    st.error("Por favor, ingrese un valor para la clase.")
    st.stop()  

st.write(f"La clase ingresada es: {pclass}")

inputs_dict = {
    "Pclass":int(class_input),
    "Age":int(age_input),	
    "SibSp":int(sibs_input),	
    "Fare":int(fare_input),	
    "sex":sex_input
}

prediction_df = pd.DataFrame(inputs_dict, index=[1])

prediction_result = lr.predict(prediction_df)

if prediction_result == 0:
    prediction_result = "No sobreviviste \U0001F622"
    st.write(prediction_result)

elif prediction_result == 1:
    prediction_result = "Sobreviviste!"
    st.write(prediction_result)

else:
    prediction_result = "Esperando a que se ingresen los datos"
    st.write(prediction_result)
