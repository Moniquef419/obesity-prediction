import numpy as np
import pandas as pd
import streamlit as st
import joblib
import altair as alt

st.set_page_config(page_title="Previsor de Obesidade", layout="wide")


def inject_styles():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

            :root {
                --med-blue-900: #0b3c6d;
                --med-blue-700: #125f9a;
                --med-blue-500: #1f84c7;
                --med-cyan-400: #2db7b6;
                --med-bg-100: #eef5fb;
                --med-bg-200: #e3eff9;
                --med-surface: #ffffff;
                --med-text: #123047;
                --med-border: #c8d9e8;
            }

            html, body, [class*="css"] {
                font-family: "Manrope", sans-serif;
                color: var(--med-text);
            }

            .stApp {
                background:
                    radial-gradient(circle at 5% 0%, #dbeafe 0%, transparent 45%),
                    radial-gradient(circle at 95% 0%, #d1fae5 0%, transparent 40%),
                    linear-gradient(180deg, var(--med-bg-100) 0%, #f8fbff 55%, #ffffff 100%);
            }

            .hero {
                background: linear-gradient(120deg, var(--med-blue-900) 0%, var(--med-blue-700) 55%, var(--med-cyan-400) 100%);
                border-radius: 18px;
                padding: 1.2rem 1.4rem;
                color: #ffffff;
                margin-bottom: 1rem;
                box-shadow: 0 14px 30px rgba(18, 95, 154, 0.25);
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
                background: var(--med-surface);
                border: 1px solid var(--med-border);
                border-left: 4px solid var(--med-blue-500);
                border-radius: 14px;
                padding: 0.9rem 1rem;
                box-shadow: 0 6px 16px rgba(16, 47, 70, 0.08);
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
                background: linear-gradient(90deg, var(--med-blue-700), var(--med-blue-500));
                box-shadow: 0 8px 18px rgba(31, 132, 199, 0.32);
            }

            .stButton > button:hover {
                filter: brightness(1.05);
            }

            div[data-baseweb="select"] > div,
            div[data-baseweb="input"] > div {
                background-color: #f8fcff;
                border-color: var(--med-border);
            }

            div[data-baseweb="input"] input,
            div[data-baseweb="select"] * {
                color: var(--med-text);
            }

            [data-testid="stForm"] {
                background: rgba(255, 255, 255, 0.86);
                border: 1px solid var(--med-border);
                border-radius: 16px;
                padding: 1rem 1rem 0.6rem 1rem;
            }

            @media (prefers-color-scheme: dark) {
                .stApp {
                    background:
                        radial-gradient(circle at 5% 0%, #102a43 0%, transparent 45%),
                        radial-gradient(circle at 95% 0%, #134e4a 0%, transparent 40%),
                        linear-gradient(180deg, #0b1220 0%, #0f172a 55%, #111827 100%);
                }
                .card {
                    background: #111827;
                    border: 1px solid #334155;
                    border-left: 4px solid #38bdf8;
                    color: #e5e7eb;
                }
                .stMarkdown, .stCaption, label, p, h1, h2, h3 {
                    color: #e5e7eb !important;
                }
                .stDataFrame, .stTable {
                    color: #e5e7eb;
                }
                [data-testid="stForm"] {
                    background: rgba(17, 24, 39, 0.75);
                    border: 1px solid #334155;
                }
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
            <strong>Classificação prevista:</strong> {label}<br/>
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
        st.error(f"Arquivo de modelo não encontrado: {exc}")
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
        st.error(f"Arquivo CSV não encontrado: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Erro ao carregar CSV: {exc}")
        st.stop()


model, le = load_model()
df = load_data()
inject_styles()

render_hero(
    "Previsor de Obesidade",
    "Suporte clínico com predição individual e painel analítico interativo.",
)

menu = st.radio("Menu:", ["Prever", "Painel Analítico"], horizontal=True)

if menu == "Prever":
    st.subheader("Cadastro do Paciente")
    st.caption("Preencha os dados clínicos e comportamentais do paciente.")

    gender_map = {"Feminino": "Female", "Masculino": "Male"}
    yes_no_map = {"Sim": "yes", "Não": "no"}
    caec_map = {
        "Não": "no",
        "Às vezes": "Sometimes",
        "Frequentemente": "Frequently",
        "Sempre": "Always",
    }
    calc_map = {
        "Não": "no",
        "Às vezes": "Sometimes",
        "Frequentemente": "Frequently",
        "Sempre": "Always",
    }
    mtrans_map = {
        "Carro": "Automobile",
        "Moto": "Motorbike",
        "Bicicleta": "Bike",
        "Transporte público": "Public_Transportation",
        "A pé": "Walking",
    }

    with st.form("form_prever", clear_on_submit=False):
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown('<div class="card"><strong>Dados Pessoais</strong></div>', unsafe_allow_html=True)
            gender_label = st.selectbox("Gênero", list(gender_map.keys()))
            age = st.number_input("Idade", min_value=14, max_value=61, value=30)
            height = st.number_input("Altura (m)", min_value=1.45, max_value=1.98, value=1.70, step=0.01)
            weight = st.number_input("Peso (kg)", min_value=39.0, max_value=173.0, value=70.0, step=0.1)
            family_history_label = st.selectbox("Histórico familiar de excesso de peso?", list(yes_no_map.keys()))
            favc_label = st.selectbox("Consumo frequente de alimentos muito calóricos?", list(yes_no_map.keys()))
            fcvc = st.selectbox(
                "Frequência de consumo de vegetais (FCVC): 1 raramente, 2 às vezes, 3 sempre",
                options=[1, 2, 3],
                index=1,
            )

        with col2:
            st.markdown('<div class="card"><strong>Hábitos de Vida</strong></div>', unsafe_allow_html=True)
            ncp = st.selectbox(
                "Número de refeições principais por dia: 1 a 4",
                options=[1, 2, 3, 4],
                index=2,
            )
            caec_label = st.selectbox(
                "Consumo de lanches entre as refeições?",
                list(caec_map.keys()),
            )
            smoke_label = st.selectbox("Hábito de fumar?", list(yes_no_map.keys()))
            ch2o = st.selectbox(
                "Consumo diário de água (CH2O): 1 < 1L, 2 1-2L, 3 > 2L",
                options=[1, 2, 3],
                index=1,
            )
            scc_label = st.selectbox("Monitora a ingestão calórica diária?", list(yes_no_map.keys()))
            faf = st.selectbox(
                "Frequência semanal de atividade física: 0 a 3",
                options=[0, 1, 2, 3],
                index=1,
            )
            tue = st.selectbox(
                "Tempo em dispositivos eletrônicos: 0 a 2",
                options=[0, 1, 2],
                index=1,
            )
            calc_label = st.selectbox("Consumo de bebida alcoólica?", list(calc_map.keys()))
            mtrans_label = st.selectbox("Meio de transporte habitual", list(mtrans_map.keys()))

        bmi_preview = weight / (height ** 2)
        st.info(f"IMC calculado automaticamente: {bmi_preview:.1f}")
        submitted = st.form_submit_button("Prever classificação")

    if submitted:
        try:
            gender = gender_map[gender_label]
            family_history = yes_no_map[family_history_label]
            favc = yes_no_map[favc_label]
            caec = caec_map[caec_label]
            smoke = yes_no_map[smoke_label]
            scc = yes_no_map[scc_label]
            calc = calc_map[calc_label]
            mtrans = mtrans_map[mtrans_label]

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
            st.error(f"Erro na previsão: {exc}")

else:
    st.subheader("Painel Analítico")
    st.caption("Painel simplificado para leitura clínica rápida.")

    with st.expander("Filtros", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender_opts = sorted(df["Gender"].dropna().unique().tolist())
            selected_gender = st.multiselect("Gênero", gender_opts, default=gender_opts)
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

    class_pt = {
        "Insufficient_Weight": "Abaixo do peso",
        "Normal_Weight": "Peso normal",
        "Overweight_Level_I": "Sobrepeso I",
        "Overweight_Level_II": "Sobrepeso II",
        "Obesity_Type_I": "Obesidade I",
        "Obesity_Type_II": "Obesidade II",
        "Obesity_Type_III": "Obesidade III",
    }

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

    age_bins = pd.cut(
        df_filtered["Age"],
        bins=[0, 18, 25, 35, 45, 60, 120],
        labels=["<=18", "19-25", "26-35", "36-45", "46-60", "60+"],
    )

    tab1, tab2, tab3 = st.tabs(["Visão Geral", "Risco", "Hábitos"])

    with tab1:
        st.markdown('<div class="card"><strong>Prevalencia por Classe (%)</strong></div>', unsafe_allow_html=True)
        class_pct = (df_filtered["Obesity"].value_counts(normalize=True) * 100).sort_values(ascending=False)
        class_pct.index = [class_pt.get(c, c) for c in class_pct.index]
        st.bar_chart(class_pct)

        st.markdown('<div class="card"><strong>IMC por Classe (Boxplot)</strong></div>', unsafe_allow_html=True)
        box_df = df_filtered[["Obesity", "BMI"]].copy()
        box_df["Classe"] = box_df["Obesity"].map(class_pt)
        boxplot = (
            alt.Chart(box_df)
            .mark_boxplot()
            .encode(
                x=alt.X("Classe:N", sort=None, title="Classe"),
                y=alt.Y("BMI:Q", title="IMC"),
                color=alt.Color("Classe:N", legend=None),
                tooltip=["Classe:N", alt.Tooltip("BMI:Q", format=".2f")],
            )
            .properties(height=320)
        )
        st.altair_chart(boxplot, use_container_width=True)

        st.markdown('<div class="card"><strong>Idade x IMC (Dispersão)</strong></div>', unsafe_allow_html=True)
        scatter_df = df_filtered[["Age", "BMI", "Obesity"]].copy()
        scatter_df["Classe"] = scatter_df["Obesity"].map(class_pt)
        scatter = (
            alt.Chart(scatter_df)
            .mark_circle(size=55, opacity=0.55)
            .encode(
                x=alt.X("Age:Q", title="Idade"),
                y=alt.Y("BMI:Q", title="IMC"),
                color=alt.Color("Classe:N", title="Classe"),
                tooltip=["Classe:N", alt.Tooltip("Age:Q", format=".0f"), alt.Tooltip("BMI:Q", format=".2f")],
            )
            .properties(height=340)
        )
        st.altair_chart(scatter, use_container_width=True)

    with tab2:
        st.markdown('<div class="card"><strong>Risco clínico alto por Gênero (%)</strong></div>', unsafe_allow_html=True)
        risk_by_gender = (
            df_filtered.assign(high_risk=df_filtered["Obesity"].isin(risk_classes))
            .groupby("Gender")["high_risk"]
            .mean()
            .mul(100)
            .rename(index={"Female": "Feminino", "Male": "Masculino"})
        )
        st.bar_chart(risk_by_gender)

        st.markdown('<div class="card"><strong>Risco clínico alto por Faixa de Idade (%)</strong></div>', unsafe_allow_html=True)
        risk_by_age = (
            df_filtered.assign(age_group=age_bins, high_risk=df_filtered["Obesity"].isin(risk_classes))
            .groupby("age_group", observed=False)["high_risk"]
            .mean()
            .mul(100)
        )
        st.bar_chart(risk_by_age)

        risk_table = pd.DataFrame(
            {
                "Grupo": list(risk_by_gender.index) + list(risk_by_age.index.astype(str)),
                "Risco Alto (%)": list(risk_by_gender.round(1).values) + list(risk_by_age.round(1).values),
            }
        )
        st.markdown('<div class="card"><strong>Tabela-resumo de risco</strong></div>', unsafe_allow_html=True)
        st.dataframe(risk_table, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown('<div class="card"><strong>Risco alto por Hábito (Sim/Não)</strong></div>', unsafe_allow_html=True)
        habits = {
            "family_history": "Histórico familiar",
            "FAVC": "Alimentos caloricos frequentes",
            "SMOKE": "Fumante",
            "SCC": "Monitora calorias",
        }
        risk_habits_rows = []
        for col, label in habits.items():
            if col not in df_filtered.columns:
                continue
            tmp = df_filtered[[col, "Obesity"]].copy()
            tmp["opt"] = tmp[col].astype(str).str.lower().map({"yes": "Sim", "no": "Não"})
            tmp = tmp.dropna(subset=["opt"])
            agg = tmp.assign(high_risk=tmp["Obesity"].isin(risk_classes)).groupby("opt")["high_risk"].mean() * 100
            for opt in ["Sim", "Não"]:
                if opt in agg.index:
                    risk_habits_rows.append(
                        {"Hábito": label, "Resposta": opt, "Risco Alto (%)": float(agg.loc[opt])}
                    )

        risk_habits_df = pd.DataFrame(risk_habits_rows)
        habits_chart = (
            alt.Chart(risk_habits_df)
            .mark_bar()
            .encode(
                x=alt.X("Hábito:N", title="Hábito"),
                y=alt.Y("Risco Alto (%):Q", title="Risco Alto (%)"),
                color=alt.Color("Resposta:N", title="Resposta"),
                xOffset="Resposta:N",
                tooltip=["Hábito:N", "Resposta:N", alt.Tooltip("Risco Alto (%):Q", format=".1f")],
            )
            .properties(height=320)
        )
        st.altair_chart(habits_chart, use_container_width=True)

        st.markdown('<div class="card"><strong>Amostra dos Dados Filtrados</strong></div>', unsafe_allow_html=True)
        st.dataframe(df_filtered.sample(min(80, len(df_filtered)), random_state=42), use_container_width=True)





