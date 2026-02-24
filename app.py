import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'obesity_pipeline.joblib')
LE_PATH = os.path.join(os.path.dirname(__file__), 'models', 'label_encoder.joblib')
DATA_PATH = os.path.join(os.path.dirname(__file__), 'Obesity.csv')

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LE_PATH)
    return model, le

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    return df

model, le = load_model()
df = load_data()

st.set_page_config(page_title="Previsor de Obesidade", layout="wide")
st.title('🏥 Previsor de Obesidade - Hospital')
st.write('Sistema de apoio à decisão clínica para diagnóstico de obesidade.')

# Menu de navegação
menu = st.radio('Selecione:', ['Prever', 'Painel Analítico'], horizontal=True)

if menu == 'Prever':
    st.header('📋 Entrada do Paciente')
    
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox('Gênero', ['Female', 'Male'])
        age = st.number_input('Idade', min_value=1, max_value=120, value=30)
        height = st.number_input('Altura (m)', min_value=0.5, max_value=2.5, value=1.7, format="%.2f")
        weight = st.number_input('Peso (kg)', min_value=10.0, max_value=300.0, value=70.0, format="%.1f")
        family_history = st.selectbox('Histórico familiar de excesso de peso?', ['yes', 'no'])
        favc = st.selectbox('Come alimentos altamente calóricos frequentemente?', ['yes', 'no'])
        fcvc = st.number_input('Consumo de vegetais (0-10)', min_value=0, max_value=10, value=2)
    
    with col2:
        ncp = st.number_input('Refeições principais por dia', min_value=1, max_value=10, value=3)
        caec = st.selectbox('Come entre refeições?', ['no', 'Sometimes', 'Frequently', 'Always'])
        smoke = st.selectbox('Fuma?', ['yes', 'no'])
        ch2o = st.number_input('Água por dia (litros)', min_value=0, max_value=20, value=2)
        scc = st.selectbox('Monitora calorias?', ['yes', 'no'])
        faf = st.number_input('Atividade física (horas/semana)', min_value=0, max_value=10, value=1)
        tue = st.number_input('Tempo com dispositivos (horas/dia)', min_value=0, max_value=24, value=1)
        calc = st.selectbox('Consumo de álcool?', ['no', 'Sometimes', 'Frequently', 'Always'])
        mtrans = st.selectbox('Meio de transporte?', ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])

    if st.button('🔍 Fazer Previsão', key='predict_btn'):
        try:
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
            
            # map binary yes/no to 1/0
            for col in ['family_history', 'FAVC', 'SMOKE', 'SCC']:
                if col in X.columns:
                    X[col] = X[col].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
            
            # ensure numeric columns
            for col in ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI']:
                if col in X.columns:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
            
            preds = model.predict(X)
            label = le.inverse_transform(preds)[0]
            
            st.success(f'### Resultado: **{label}**')
            
            try:
                probs = model.predict_proba(X)
                top_idx = np.argsort(probs[0])[::-1][:3]
                st.write('**Probabilidades (top 3):**')
                for idx in top_idx:
                    st.write(f'- {le.inverse_transform([idx])[0]}: {probs[0][idx]:.1%}')
            except:
                pass
        except Exception as e:
            st.error(f'Erro na previsão: {str(e)}')

elif menu == 'Painel Analítico':
    st.header('📊 Análise de Dados')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Distribuição de Obesidade')
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        df['Obesity'].value_counts().plot(kind='barh', ax=ax1, color='steelblue')
        ax1.set_xlabel('Quantidade')
        st.pyplot(fig1, use_container_width=True)
    
    with col2:
        st.subheader('Distribuição de BMI por Classe')
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        df.boxplot(column='BMI', by='Obesity', ax=ax2)
        plt.suptitle('')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig2, use_container_width=True)
    
    st.subheader('Idade vs BMI')
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    for obesity_class in df['Obesity'].unique():
        mask = df['Obesity'] == obesity_class
        ax3.scatter(df[mask]['Age'], df[mask]['BMI'], label=obesity_class, alpha=0.6)
    ax3.set_xlabel('Idade')
    ax3.set_ylabel('BMI')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig3, use_container_width=True)
    
    st.subheader('Amostra dos Dados')
    st.dataframe(df.sample(100, random_state=42).reset_index(drop=True), use_container_width=True)

