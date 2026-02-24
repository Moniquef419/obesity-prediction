import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'obesity_pipeline.joblib')
LE_PATH = os.path.join(os.path.dirname(__file__), 'models', 'label_encoder.joblib')
DATA_PATH = os.path.join(os.path.dirname(__file__), 'Obesity.csv')

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LE_PATH)
    return model, le

model, le = load_model()

st.title('Previsor de Obesidade - Hospital')
st.write('Interface para prever o nível de obesidade a partir das características do paciente.')

tabs = st.tabs(['Prever', 'Painel Analítico'])

with tabs[0]:
    st.header('Entrada do paciente')
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox('Gênero', ['Female', 'Male'])
        age = st.number_input('Idade', min_value=1, max_value=120, value=30)
        height = st.number_input('Altura (m)', min_value=0.5, max_value=2.5, value=1.7, format="%.2f")
        weight = st.number_input('Peso (kg)', min_value=10.0, max_value=300.0, value=70.0, format="%.1f")
        family_history = st.radio('Histórico familiar de excesso de peso?', ['yes', 'no'])
        favc = st.radio('Come alimentos altamente calóricos frequentemente?', ['yes', 'no'])
        fcvc = st.number_input('Quantas porções de vegetais nas refeições (FCVC)', min_value=0, max_value=10, value=2)
    with col2:
        ncp = st.number_input('Número de refeições principais por dia (NCP)', min_value=1, max_value=10, value=3)
        caec = st.selectbox('Come entre as refeições (CAEC)', ['no', 'Sometimes', 'Frequently', 'Always'])
        smoke = st.radio('Fuma?', ['yes', 'no'])
        ch2o = st.number_input('Quantidade de água por dia (CH2O)', min_value=0, max_value=20, value=2)
        scc = st.radio('Monitora calorias (SCC)?', ['yes', 'no'])
        faf = st.number_input('Frequência de atividade física (FAF)', min_value=0, max_value=10, value=1)
        tue = st.number_input('Tempo de uso de dispositivos (TUE)', min_value=0, max_value=24, value=1)
        calc = st.selectbox('Frequência de consumo de álcool (CALC)', ['no', 'Sometimes', 'Frequently', 'Always'])
        mtrans = st.selectbox('Meio de transporte (MTRANS)', ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])

    if st.button('Prever'):
        bmi = weight / (height ** 2)
        data = {
            'Gender': gender,
            'Age': age,
            'family_history': family_history,
            'FAVC': favc,
            'FCVC': fcvc,
            'NCP': ncp,
            'CAEC': caec,
            'SMOKE': smoke,
            'CH2O': ch2o,
            'SCC': scc,
            'FAF': faf,
            'TUE': tue,
            'CALC': calc,
            'MTRANS': mtrans,
            'BMI': bmi,
        }
        X = pd.DataFrame([data])
        # map binary yes/no to 1/0 to match training preprocessing
        for col in ['family_history', 'FAVC', 'SMOKE', 'SCC']:
            if col in X.columns:
                X[col] = X[col].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})

        # ensure numeric columns are proper dtype
        for col in ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI']:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        # model expects the same feature columns used in training
        preds = model.predict(X)
        probs = None
        try:
            probs = model.predict_proba(X)
        except Exception:
            pass
        label = le.inverse_transform(preds)[0]
        st.subheader('Previsão:')
        st.write(f'**{label}**')
        if probs is not None:
            top_idx = np.argsort(probs[0])[::-1][:3]
            st.write('Probabilidades (top 3):')
            for idx in top_idx:
                st.write(f'{le.inverse_transform([idx])[0]}: {probs[0][idx]:.3f}')

with tabs[1]:
    st.header('Painel Analítico')
    df = pd.read_csv(DATA_PATH)
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)

    st.subheader('Distribuição do alvo (Obesity)')
    fig1, ax1 = plt.subplots(figsize=(8,4))
    sns.countplot(data=df, y='Obesity', order=df['Obesity'].value_counts().index, palette='viridis', ax=ax1)
    ax1.set_xlabel('Count')
    st.pyplot(fig1)

    st.subheader('Distribuição de BMI por categoria')
    fig2, ax2 = plt.subplots(figsize=(8,4))
    sns.boxplot(data=df, x='Obesity', y='BMI', palette='viridis', ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.subheader('Relação: Idade vs BMI (color por Obesity)')
    fig3, ax3 = plt.subplots(figsize=(8,5))
    sns.scatterplot(data=df, x='Age', y='BMI', hue='Obesity', alpha=0.7, ax=ax3)
    st.pyplot(fig3)

    st.markdown('### Amostra dos dados')
    st.dataframe(df.sample(200, random_state=42).reset_index(drop=True))
