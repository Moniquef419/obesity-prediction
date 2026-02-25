import numpy as np
import pandas as pd
import streamlit as st
import joblib


st.set_page_config(page_title="Previsor de Obesidade", layout="wide")


def inject_styles():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

            html, body, [class*="css"] {
                font-family: "Manrope", sans-serif;
            }

            .stApp {
                background: radial-gradient(circle at 10% 0%, #eaf7f5 0%, #f7fafc 35%, #ffffff 100%);
            }

            .hero {
                background: linear-gradient(120deg, #0f766e 0%, #155e75 45%, #0e7490 100%);
                border-radius: 18px;
                padding: 1.2rem 1.4rem;
                color: #ffffff;
                margin-bottom: 1rem;
                box-shadow: 0 12px 30px rgba(15, 118, 110, 0.22);
            }

            .hero h1 {
                margin: 0;
                font-size: 1.8rem;
                font-weight: 800;
                letter-spacing: -0.02em;
            }

            .hero p {
                margin: 0.4rem 0 0;
                opacity: 0.95;
                font-weight: 500;
            }

            .card {
                background: #ffffff;
                border: 1px solid #dbe5ec;
                border-radius: 14px;
                padding: 0.9rem 1rem;
                box-shadow: 0 5px 14px rgba(15, 23, 42, 0.05);
                margin-bottom: 0.8rem;
            }

            .risk-chip {
                display: inline-block;
                padding: 0.4rem 0.7rem;
                border-radius: 999px;
                font-size: 0.84rem;
                font-weight: 700;
                margin-top: 0.4rem;
            }

            .risk-low { background: #dcfce7; color: #166534; }
            .risk-med { background: #fef9c3; color: #854d0e; }
            .risk-high { background: #fee2e2; color: #991b1b; }

            .stButton > button {
                width: 100%;
                border: 0;
                border-radius: 10px;
                color: white;
                font-weight: 700;
                background: linear-gradient(90deg, #0f766e, #0e7490);
                box-shadow: 0 8px 18px rgba(14, 116, 144, 0.28);
            }

            .stButton > button:hover {
                filter: brightness(1.05);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_badge(label):
    high = {"Obesity_Type_II", "Obesity_Type_III"}
    medium = {"Overweight_Level_I", "Overweight_Level_II", "Obesity_Type_I"}
    if label in high:
        css_class = "risk-chip risk-high"
        text = "Risco Elevado"
    elif label in medium:
        css_class = "risk-chip risk-med"
        text = "Risco Moderado"
    else:
        css_class = "risk-chip risk-low"
        text = "Risco Baixo"

    st.markdown(
        f"""
        <div class="card">
            <strong>Classificacao prevista:</strong> {label}<br/>
            <span class="{css_class}">{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
inject_styles()

render_hero(
    "Previsor de Obesidade",
    "Suporte clinico com predicao individual e painel analitico interativo.",
)

menu = st.radio("Menu:", ["Prever", "Painel Analitico"], horizontal=True)

if menu == "Prever":
    st.subheader("Cadastro do Paciente")
    col1, col2 = st.columns(2)
    st.caption(
        "Os campos abaixo seguem o dicionario oficial: codigo, descricao e valores validos."
    )

    with col1:
        gender = st.selectbox(
            "Gender — (Genero) Sexo biologico. Valores: Female, Male.",
            ["Female", "Male"],
        )
        age = st.number_input(
            "Age — (Idade) Idade em anos. Valores: numerico continuo (min 14, max 61).",
            min_value=14,
            max_value=61,
            value=30,
        )
        height = st.number_input(
            "Height — (Altura) Altura em metros. Valores: numerico continuo (ex.: 1.45-1.98 m).",
            min_value=1.45,
            max_value=1.98,
            value=1.70,
            step=0.01,
        )
        weight = st.number_input(
            "Weight — (Peso) Peso em quilogramas. Valores: numerico continuo (ex.: 39-173 kg).",
            min_value=39.0,
            max_value=173.0,
            value=70.0,
            step=0.1,
        )
        family_history = st.selectbox(
            "family_history — (Family history of overweight) Historico familiar de excesso de peso. Valores: yes, no.",
            ["yes", "no"],
        )
        favc = st.selectbox(
            "FAVC — (Frequent consumption of high-caloric food) Consumo frequente de alimentos muito caloricos. Valores: yes, no.",
            ["yes", "no"],
        )
        fcvc = st.selectbox(
            "FCVC — (Frequency of consumption of vegetables) Frequencia de consumo de vegetais nas refeicoes. "
            "Valores (escala 1-3): 1 raramente, 2 as vezes, 3 sempre.",
            options=[1, 2, 3],
            index=1,
        )

    with col2:
        ncp = st.selectbox(
            "NCP — (Number of main meals) Numero de refeicoes principais por dia. "
            "Valores (escala 1-4): 1 uma refeicao, 2 duas, 3 tres, 4 quatro ou mais.",
            options=[1, 2, 3, 4],
            index=2,
        )
        caec = st.selectbox(
            "CAEC — (Consumption of food between meals) Consumo de lanches/comes entre as refeicoes. "
            "Valores: no, Sometimes, Frequently, Always.",
            ["no", "Sometimes", "Frequently", "Always"],
        )
        smoke = st.selectbox(
            "SMOKE — (Smoking) Habito de fumar. Valores: yes, no.",
            ["yes", "no"],
        )
        ch2o = st.selectbox(
            "CH2O — (Daily water consumption) Consumo diario de agua. "
            "Valores (escala 1-3): 1 < 1 L/dia, 2 1-2 L/dia, 3 > 2 L/dia.",
            options=[1, 2, 3],
            index=1,
        )
        scc = st.selectbox(
            "SCC — (Calories consumption monitoring) Monitora a ingestao calorica diaria. Valores: yes, no.",
            ["yes", "no"],
        )
        faf = st.selectbox(
            "FAF — (Physical activity frequency) Frequencia semanal de atividade fisica. "
            "Valores (escala 0-3): 0 nenhuma, 1 ~1-2x/sem, 2 ~3-4x/sem, 3 5x/sem ou mais.",
            options=[0, 1, 2, 3],
            index=1,
        )
        tue = st.selectbox(
            "TUE — (Time using electronic devices) Tempo diario usando dispositivos eletronicos. "
            "Valores (escala 0-2): 0 ~0-2 h/dia, 1 ~3-5 h/dia, 2 > 5 h/dia.",
            options=[0, 1, 2],
            index=1,
        )
        calc = st.selectbox(
            "CALC — (Alcohol consumption) Consumo de bebida alcoolica. Valores: no, Sometimes, Frequently, Always.",
            ["no", "Sometimes", "Frequently", "Always"],
        )
        mtrans = st.selectbox(
            "MTRANS — (Transportation used / Mode of transport) Meio de transporte habitual. "
            "Valores: Automobile, Motorbike, Bike, Public_Transportation, Walking.",
            ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"],
        )

    if st.button("Prever"):
        try:
            fcvc = float(np.clip(np.rint(fcvc), 1, 3))
            ncp = float(np.clip(np.rint(ncp), 1, 4))
            ch2o = float(np.clip(np.rint(ch2o), 1, 3))
            faf = float(np.clip(np.rint(faf), 0, 3))
            tue = float(np.clip(np.rint(tue), 0, 2))

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
            render_result_badge(label)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                top_idx = np.argsort(probs)[::-1][:3]
                st.markdown('<div class="card"><strong>Top 3 probabilidades</strong></div>', unsafe_allow_html=True)
                pcol1, pcol2, pcol3 = st.columns(3)
                for pos, idx in enumerate(top_idx):
                    class_name = le.inverse_transform([idx])[0]
                    prob_txt = f"{probs[idx]:.1%}"
                    if pos == 0:
                        pcol1.metric(class_name, prob_txt)
                    elif pos == 1:
                        pcol2.metric(class_name, prob_txt)
                    else:
                        pcol3.metric(class_name, prob_txt)
        except Exception as exc:
            st.error(f"Erro na previsao: {exc}")

else:
    st.subheader("Painel Analitico")
    st.caption("Visao para apoiar decisao clinica com foco em risco e perfil de pacientes.")

    with st.expander("Filtros", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender_opts = sorted(df["Gender"].dropna().unique().tolist())
            selected_gender = st.multiselect("Genero", gender_opts, default=gender_opts)
        with c2:
            age_min = int(df["Age"].min())
            age_max = int(df["Age"].max())
            selected_age = st.slider("Faixa de idade", age_min, age_max, (age_min, age_max))
        with c3:
            class_opts = sorted(df["Obesity"].dropna().unique().tolist())
            selected_class = st.multiselect("Classe de obesidade", class_opts, default=class_opts)

    df_filtered = df[
        (df["Gender"].isin(selected_gender))
        & (df["Age"].between(selected_age[0], selected_age[1]))
        & (df["Obesity"].isin(selected_class))
    ].copy()

    if df_filtered.empty:
        st.warning("Sem dados para os filtros selecionados.")
        st.stop()

    risk_classes = [
        "Overweight_Level_II",
        "Obesity_Type_I",
        "Obesity_Type_II",
        "Obesity_Type_III",
    ]
    severe_rate = (df_filtered["Obesity"].isin(risk_classes).mean()) * 100
    excess_weight_rate = (
        df_filtered["Obesity"].str.contains("Overweight|Obesity", regex=True).mean() * 100
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pacientes analisados", f"{len(df_filtered):,}".replace(",", "."))
    m2.metric("BMI medio", f"{df_filtered['BMI'].mean():.1f}")
    m3.metric("Excesso de peso", f"{excess_weight_rate:.1f}%")
    m4.metric("Risco clinico alto", f"{severe_rate:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><strong>Distribuicao das Classes (%)</strong></div>', unsafe_allow_html=True)
        class_pct = (df_filtered["Obesity"].value_counts(normalize=True) * 100).sort_values(ascending=False)
        st.bar_chart(class_pct)
    with col2:
        st.markdown('<div class="card"><strong>BMI medio por Classe</strong></div>', unsafe_allow_html=True)
        bmi_by_class = df_filtered.groupby("Obesity")["BMI"].mean().sort_values(ascending=False)
        st.bar_chart(bmi_by_class)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="card"><strong>BMI medio por Faixa de Idade</strong></div>', unsafe_allow_html=True)
        age_bins = pd.cut(
            df_filtered["Age"],
            bins=[0, 18, 25, 35, 45, 60, 120],
            labels=["<=18", "19-25", "26-35", "36-45", "46-60", "60+"],
        )
        bmi_age = df_filtered.assign(age_group=age_bins).groupby("age_group", observed=False)["BMI"].mean()
        st.line_chart(bmi_age)
    with col4:
        st.markdown('<div class="card"><strong>Distribuicao por Genero e Classe</strong></div>', unsafe_allow_html=True)
        gender_class = pd.crosstab(df_filtered["Gender"], df_filtered["Obesity"])
        st.bar_chart(gender_class)

    st.markdown('<div class="card"><strong>Indicadores de Habitos (% yes)</strong></div>', unsafe_allow_html=True)
    habits = ["family_history", "FAVC", "SMOKE", "SCC"]
    habits_yes = pd.Series(
        {
            col: (df_filtered[col].astype(str).str.lower() == "yes").mean() * 100
            for col in habits
            if col in df_filtered.columns
        }
    ).sort_values(ascending=False)
    st.bar_chart(habits_yes)

    st.markdown('<div class="card"><strong>Amostra dos Dados Filtrados</strong></div>', unsafe_allow_html=True)
    st.dataframe(df_filtered.sample(min(100, len(df_filtered)), random_state=42), use_container_width=True)
