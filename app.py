import numpy as np
import pandas as pd
import streamlit as st
import joblib


st.set_page_config(page_title="Previsor de Obesidade", layout="wide")
st.title("Previsor de Obesidade")


def load_model():
    try:
        model = joblib.load("models/obesity_pipeline.joblib")
        label_encoder = joblib.load("models/label_encoder.joblib")
        return model, label_encoder
    except FileNotFoundError as exc:
        st.error(f"Arquivo de modelo nao encontrado: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Erro ao carregar modelo: {exc}")
        st.stop()


def load_data():
    try:
        data = pd.read_csv("Obesity.csv")
        data["BMI"] = data["Weight"] / (data["Height"] ** 2)
        return data
    except FileNotFoundError as exc:
        st.error(f"Arquivo CSV nao encontrado: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Erro ao carregar CSV: {exc}")
        st.stop()


model, le = load_model()
df = load_data()

menu = st.radio("Menu:", ["Prever", "Painel Analitico"], horizontal=True)

if menu == "Prever":
    st.header("Dados do Paciente")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Genero", ["Female", "Male"])
        age = st.number_input("Idade", min_value=1, max_value=120, value=30)
        height = st.number_input("Altura (m)", min_value=0.5, max_value=2.5, value=1.7)
        weight = st.number_input("Peso (kg)", min_value=10.0, max_value=300.0, value=70.0)
        family_history = st.selectbox("Historico familiar?", ["yes", "no"])
        favc = st.selectbox("Alimentos caloricos frequentes?", ["yes", "no"])
        fcvc = st.number_input("Consumo de vegetais (FCVC)", min_value=0.0, max_value=10.0, value=2.0)

    with col2:
        ncp = st.number_input("Refeicoes por dia (NCP)", min_value=1.0, max_value=10.0, value=3.0)
        caec = st.selectbox("Come entre refeicoes (CAEC)?", ["no", "Sometimes", "Frequently", "Always"])
        smoke = st.selectbox("Fuma (SMOKE)?", ["yes", "no"])
        ch2o = st.number_input("Agua por dia (CH2O)", min_value=0.0, max_value=20.0, value=2.0)
        scc = st.selectbox("Monitora calorias (SCC)?", ["yes", "no"])
        faf = st.number_input("Atividade fisica (FAF)", min_value=0.0, max_value=10.0, value=1.0)
        tue = st.number_input("Tempo em dispositivos (TUE)", min_value=0.0, max_value=24.0, value=1.0)
        calc = st.selectbox("Consumo de alcool (CALC)", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox(
            "Meio de transporte (MTRANS)",
            ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"],
        )

    if st.button("Prever"):
        try:
            bmi = weight / (height ** 2)
            X = pd.DataFrame(
                [
                    {
                        "Gender": gender,
                        "Age": age,
                        "family_history": family_history,
                        "FAVC": favc,
                        "FCVC": fcvc,
                        "NCP": ncp,
                        "CAEC": caec,
                        "SMOKE": smoke,
                        "CH2O": ch2o,
                        "SCC": scc,
                        "FAF": faf,
                        "TUE": tue,
                        "CALC": calc,
                        "MTRANS": mtrans,
                        "BMI": bmi,
                    }
                ]
            )

            for col in ["family_history", "FAVC", "SMOKE", "SCC"]:
                X[col] = X[col].map({"yes": 1, "no": 0})

            for col in ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE", "BMI"]:
                X[col] = pd.to_numeric(X[col], errors="coerce")

            pred = model.predict(X)[0]
            label = le.inverse_transform([pred])[0]
            st.success(f"Resultado: {label}")

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                top_idx = np.argsort(probs)[::-1][:3]
                st.write("Top 3 probabilidades:")
                for idx in top_idx:
                    class_name = le.inverse_transform([idx])[0]
                    st.write(f"- {class_name}: {probs[idx]:.1%}")
        except Exception as exc:
            st.error(f"Erro na previsao: {exc}")

else:
    st.header("Painel Analitico")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribuicao das Classes")
        st.bar_chart(df["Obesity"].value_counts())

    with col2:
        st.subheader("BMI por Classe")
        bmi_by_class = df.groupby("Obesity")["BMI"].mean().sort_values(ascending=False)
        st.bar_chart(bmi_by_class)

    st.subheader("Amostra de Dados")
    st.dataframe(df.sample(min(100, len(df)), random_state=42), use_container_width=True)
