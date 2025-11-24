import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Cargar modelo y encoder
model = joblib.load("iris_model.pkl")
encoder = joblib.load("encoder.pkl")

# Cargar dataset
df = pd.read_csv("Iris.csv")
df = df.drop(columns=["Id"], errors="ignore")

st.title("🌸 Iris Species Classification Dashboard")

# Sidebar inputs
st.sidebar.header("Ingrese las características de la flor")
sepal_length = st.sidebar.number_input("Sepal Length (cm)", 0.0, 10.0, 5.0)
sepal_width = st.sidebar.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
petal_length = st.sidebar.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
petal_width = st.sidebar.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

# Predicción
if st.sidebar.button("Predecir especie"):
    new_data = pd.DataFrame({
        "SepalLengthCm": [sepal_length],
        "SepalWidthCm": [sepal_width],
        "PetalLengthCm": [petal_length],
        "PetalWidthCm": [petal_width],
    })

    pred = model.predict(new_data)
    pred_species = encoder.inverse_transform(pred)[0]

    st.success(f"🌼 La especie predicha es: **{pred_species}**")

    # Gráfico 3D con la nueva flor
    fig = px.scatter_3d(
        df,
        x="PetalLengthCm",
        y="PetalWidthCm",
        z="SepalLengthCm",
        color="Species",
        title="Posición de la flor en el dataset"
    )

    fig.add_scatter3d(
        x=[petal_length],
        y=[petal_width],
        z=[sepal_length],
        mode="markers",
        marker=dict(size=10, color="black"),
        name="Nueva flor"
    )

    st.plotly_chart(fig)
