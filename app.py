import streamlit as st
import pandas as pd
import joblib

try:
    model = joblib.load('models/obesity_pipeline.joblib')
    le = joblib.load('models/label_encoder.joblib')
    df = pd.read_csv('Obesity.csv')
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
except Exception as e:
    st.error(f"Erro ao carregar arquivos: {e}")
    st.stop()

st.set_page_config(page_title="Previsor de Obesidade", layout="wide")
st.title('🏥 Previsor de Obesidade')

menu = st.radio('Menu:', ['Prever', 'Dados'], horizontal=True)

if menu == 'Prever':
    st.header('Dados do Paciente')
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox('Gênero', ['Female', 'Male'])
        age = st.number_input('Idade', 1, 120, 30)
        height = st.number_input('Altura (m)', 0.5, 2.5, 1.7)
        weight = st.number_input('Peso (kg)', 10.0, 300.0, 70.0)
        family_history = st.selectbox('Histórico familiar?', ['yes', 'no'])
        favc = st.selectbox('Alimentos calóricos?', ['yes', 'no'])
        fcvc = st.number_input('Vegetais', 0, 10, 2)
    
    with col2:
        ncp = st.number_input('Refeições/dia', 1, 10, 3)
        caec = st.selectbox('Come entre refeições?', ['no', 'Sometimes', 'Frequently', 'Always'])
        smoke = st.selectbox('Fuma?', ['yes', 'no'])
        ch2o = st.number_input('Água/dia', 0, 20, 2)
        scc = st.selectbox('Monitora calorias?', ['yes', 'no'])
        faf = st.number_input('Atividade física', 0, 10, 1)
        tue = st.number_input('Tempo dispositivos', 0, 24, 1)
        calc = st.selectbox('Álcool?', ['no', 'Sometimes', 'Frequently', 'Always'])
        mtrans = st.selectbox('Transporte?', ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])
    
    if st.button('Prever'):
        try:
            bmi = weight / (height ** 2)
            X = pd.DataFrame([{
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
            }])
            
            for col in ['family_history', 'FAVC', 'SMOKE', 'SCC']:
                X[col] = X[col].map({'yes': 1, 'no': 0})
            
            for col in ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI']:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            
            pred = model.predict(X)[0]
            label = le.inverse_transform([pred])[0]
            st.success(f'**Resultado: {label}**')
        except Exception as e:
            st.error(f'Erro: {e}')

else:
    st.header('Amostra de Dados')
    st.dataframe(df.sample(100, random_state=42), use_container_width=True)
    st.write(f'**Total de amostras:** {len(df)}')
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


