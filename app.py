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
            @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600;700&display=swap');

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
                font-family: "Inter", sans-serif;
                color: var(--med-text);
            }

            .stApp {
                background: #f3f4f6;
            }

            .main .block-container {
                max-width: 1360px;
                padding-top: 0;
                padding-left: 2rem;
                padding-right: 2rem;
            }

            .hero {
                margin: 0 -2rem 2.2rem -2rem;
                background-image:
                    linear-gradient(90deg, rgba(11, 60, 109, 0.92) 0%, rgba(18, 95, 154, 0.75) 48%, rgba(255, 255, 255, 0.85) 100%),
                    url('https://source.unsplash.com/1920x1080/?hospital,exercise,health');
                background-size: cover;
                background-position: center;
                border-radius: 0;
                min-height: 320px;
                display: flex;
                align-items: center;
                padding: 2.2rem 3rem;
                color: #ffffff;
                box-shadow: 0 12px 24px rgba(18, 95, 154, 0.18);
            }

            .hero h1 {
                margin: 0;
                font-family: "Playfair Display", serif;
                font-size: 4rem;
                font-weight: 700;
                letter-spacing: -0.02em;
                max-width: 640px;
                line-height: 1.05;
            }

            .hero p {
                margin: 1rem 0 0;
                opacity: 0.95;
                font-weight: 400;
                font-size: 1.15rem;
                max-width: 620px;
            }

            .card {
                background: var(--med-surface);
                border: 1px solid var(--med-border);
                border-left: 4px solid var(--med-blue-500);
                border-radius: 14px;
                padding: 1.1rem 1.2rem;
                box-shadow: 0 6px 16px rgba(16, 47, 70, 0.08);
                margin-bottom: 1rem;
            }
            .card-blue { border-left-color: #2563eb; }
            .card-green { border-left-color: #16a34a; }
            .card-orange { border-left-color: #ea580c; }
            .block-header strong {
                font-family: "Playfair Display", serif;
                font-size: 1.7rem;
                font-weight: 700;
                line-height: 1.1;
            }
            .activity-header strong {
                font-size: 1.8rem;
            }
            .habits-header strong {
                font-size: 1.8rem;
            }
            .personal-header strong {
                font-size: 1.8rem;
            }

            .section-title {
                margin: 0 0 0.35rem 0;
                font-family: "Playfair Display", serif;
                font-size: 1.7rem;
                font-weight: 700;
                color: #0f1f35;
            }

            .section-subtitle {
                margin: 0 0 1.1rem 0;
                color: #5f6e7f;
                font-size: 0.98rem;
            }

            h2, h3, h4 {
                font-family: "Playfair Display", serif;
                letter-spacing: -0.01em;
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

            [data-testid="stFormSubmitButton"] > button {
                height: 54px;
                font-size: 1.2rem;
                font-weight: 700;
                border-radius: 12px;
                color: #ffffff !important;
                background: linear-gradient(90deg, #3b82f6, #2563eb);
                box-shadow: 0 8px 18px rgba(59, 130, 246, 0.24);
                border: 1px solid rgba(37, 99, 235, 0.35);
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
                padding: 1.1rem 1.2rem 0.8rem 1.2rem;
            }

            .result-panel {
                background: #ffffff;
                border: 1px solid #d1d5db;
                border-top: 4px solid #2563eb;
                border-radius: 14px;
                padding: 1.5rem;
                box-shadow: 0 6px 18px rgba(15, 23, 42, 0.10);
                position: sticky;
                top: 1.2rem;
            }

            .result-empty {
                text-align: center;
                color: #6b7280;
                padding: 2.5rem 1rem;
            }

            @media (max-width: 1100px) {
                .hero {
                    margin-left: -1rem;
                    margin-right: -1rem;
                    padding: 1.6rem 1.2rem;
                    min-height: 260px;
                }
                .hero h1 {
                    font-size: 2.6rem;
                }
            }

            @media (prefers-color-scheme: dark) {
                .stApp {
                    background: #111827;
                }
                .card {
                    background: #111827;
                    border: 1px solid #334155;
                    border-left: 4px solid #38bdf8;
                    color: #e5e7eb;
                }
                .result-panel {
                    background: #111827;
                    border: 1px solid #334155;
                    border-top: 4px solid #60a5fa;
                    color: #e5e7eb;
                }
                .stMarkdown, .stCaption, label, p, h1, h2, h3 {
                    color: #e5e7eb !important;
                }
                .section-title {
                    color: #e5e7eb;
                }
                .section-subtitle {
                    color: #cbd5e1;
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
            <div>
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_badge(label_raw, label_display):
    high = {"Obesity_Type_II", "Obesity_Type_III"}
    medium = {"Overweight_Level_I", "Overweight_Level_II", "Obesity_Type_I"}
    if label_raw in high:
        css_class = "risk-chip risk-high"
        text = "Risco Elevado"
    elif label_raw in medium:
        css_class = "risk-chip risk-med"
        text = "Risco Moderado"
    else:
        css_class = "risk-chip risk-low"
        text = "Risco Baixo"

    st.markdown(
        f"""
        <div class="card">
            <strong>Classificação prevista:</strong> {label_display}<br/>
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
    st.markdown('<h2 class="section-title">Cadastro do Paciente</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-subtitle">Preencha as informações clínicas e comportamentais do paciente.</p>',
        unsafe_allow_html=True,
    )

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

    class_pt = {
        "Insufficient_Weight": "Abaixo do peso",
        "Normal_Weight": "Peso normal",
        "Overweight_Level_I": "Sobrepeso I",
        "Overweight_Level_II": "Sobrepeso II",
        "Obesity_Type_I": "Obesidade I",
        "Obesity_Type_II": "Obesidade II",
        "Obesity_Type_III": "Obesidade III",
    }

    if "pred_result" not in st.session_state:
        st.session_state.pred_result = None

    form_col, result_col = st.columns([1.55, 0.85], gap="large")

    with form_col:
        if st.button("Limpar formulário"):
            for k in [
                "p_gender", "p_age", "p_height", "p_weight", "p_family_history",
                "p_favc", "p_fcvc", "p_ncp", "p_caec", "p_smoke", "p_ch2o",
                "p_scc", "p_faf", "p_tue", "p_calc", "p_mtrans",
            ]:
                if k in st.session_state:
                    del st.session_state[k]
            st.session_state.pred_result = None
            st.rerun()

        with st.form("form_prever", clear_on_submit=True):
            fcvc_opts = {
                "Selecione...": None,
                "1 - Raramente": 1,
                "2 - Às vezes": 2,
                "3 - Sempre": 3,
            }
            ncp_opts = {
                "Selecione...": None,
                "1 - Uma refeição": 1,
                "2 - Duas refeições": 2,
                "3 - Três refeições": 3,
                "4 - Quatro ou mais": 4,
            }
            ch2o_opts = {
                "Selecione...": None,
                "1 - Menos de 1L": 1,
                "2 - 1 a 2L": 2,
                "3 - Mais de 2L": 3,
            }
            faf_opts = {
                "Selecione...": None,
                "0 - Nenhuma": 0,
                "1 - 1 a 2 vezes": 1,
                "2 - 3 a 4 vezes": 2,
                "3 - 5 vezes ou mais": 3,
            }
            tue_opts = {
                "Selecione...": None,
                "0 - 0 a 2 horas": 0,
                "1 - 3 a 5 horas": 1,
                "2 - Mais de 5 horas": 2,
            }
            gender_opts = ["Selecione..."] + list(gender_map.keys())
            yes_no_opts = ["Selecione..."] + list(yes_no_map.keys())
            caec_opts = ["Selecione..."] + list(caec_map.keys())
            calc_opts = ["Selecione..."] + list(calc_map.keys())
            mtrans_opts = ["Selecione..."] + list(mtrans_map.keys())

            st.markdown('<div class="card card-blue block-header personal-header"><strong>Dados Pessoais</strong></div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2, gap="large")
            with c1:
                gender_label = st.selectbox("Gênero", gender_opts, index=0, key="p_gender")
                height = st.number_input("Altura (m)", min_value=1.45, max_value=1.98, value=1.70, step=0.01, key="p_height")
            with c2:
                age = st.number_input("Idade (anos)", min_value=14, max_value=61, value=30, key="p_age")
                weight = st.number_input("Peso (kg)", min_value=39.0, max_value=173.0, value=70.0, step=0.1, key="p_weight")

            st.markdown('<div class="card card-green block-header habits-header"><strong>Hábitos de Vida</strong></div>', unsafe_allow_html=True)
            family_history_label = st.selectbox("Histórico familiar de excesso de peso?", yes_no_opts, index=0, key="p_family_history")
            favc_label = st.selectbox("Consumo frequente de alimentos muito calóricos?", yes_no_opts, index=0, key="p_favc")
            fcvc_label = st.selectbox("Frequência de consumo de vegetais (FCVC)", list(fcvc_opts.keys()), index=0, key="p_fcvc")
            ncp_label = st.selectbox("Número de refeições principais por dia (NCP)", list(ncp_opts.keys()), index=0, key="p_ncp")
            caec_label = st.selectbox("Consumo de lanches entre as refeições (CAEC)?", caec_opts, index=0, key="p_caec")
            smoke_label = st.selectbox("Hábito de fumar?", yes_no_opts, index=0, key="p_smoke")
            ch2o_label = st.selectbox("Consumo diário de água (CH2O)", list(ch2o_opts.keys()), index=0, key="p_ch2o")
            scc_label = st.selectbox("Monitora a ingestão calórica diária?", yes_no_opts, index=0, key="p_scc")

            st.markdown('<div class="card card-orange block-header activity-header"><strong>Atividade Física</strong></div>', unsafe_allow_html=True)
            faf_label = st.selectbox("Frequência semanal de atividade física", list(faf_opts.keys()), index=0, key="p_faf")
            tue_label = st.selectbox("Tempo em dispositivos eletrônicos", list(tue_opts.keys()), index=0, key="p_tue")
            calc_label = st.selectbox("Consumo de bebida alcoólica?", calc_opts, index=0, key="p_calc")
            mtrans_label = st.selectbox("Meio de transporte habitual", mtrans_opts, index=0, key="p_mtrans")

            bmi_preview = weight / (height ** 2)
            st.info(f"IMC calculado automaticamente: {bmi_preview:.1f} kg/m²")
            b1, b2, b3 = st.columns([1, 3, 1])
            with b2:
                submitted = st.form_submit_button("Prever Classificação")

        if submitted:
            try:
                required = [
                    gender_label, family_history_label, favc_label, fcvc_label, ncp_label,
                    caec_label, smoke_label, ch2o_label, scc_label, faf_label, tue_label,
                    calc_label, mtrans_label,
                ]
                if "Selecione..." in required:
                    st.warning("Preencha todos os campos antes de prever.")
                    st.stop()

                gender = gender_map[gender_label]
                family_history = yes_no_map[family_history_label]
                favc = yes_no_map[favc_label]
                caec = caec_map[caec_label]
                smoke = yes_no_map[smoke_label]
                scc = yes_no_map[scc_label]
                calc = calc_map[calc_label]
                mtrans = mtrans_map[mtrans_label]
                fcvc = fcvc_opts[fcvc_label]
                ncp = ncp_opts[ncp_label]
                ch2o = ch2o_opts[ch2o_label]
                faf = faf_opts[faf_label]
                tue = tue_opts[tue_label]

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
                label_pt = class_pt.get(label, label)

                probs_top = []
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)[0]
                    top_idx = np.argsort(probs)[::-1][:3]
                    for idx in top_idx:
                        class_name = le.inverse_transform([idx])[0]
                        probs_top.append((class_pt.get(class_name, class_name), probs[idx]))

                st.session_state.pred_result = {
                    "label_raw": label,
                    "label_pt": label_pt,
                    "bmi": bmi,
                    "probs_top": probs_top,
                }
            except Exception as exc:
                st.error(f"Erro na previsão: {exc}")

    with result_col:
        pred_result = st.session_state.pred_result
        if pred_result:
            st.markdown('<div class="result-panel"><strong>Resultado</strong></div>', unsafe_allow_html=True)
            render_result_badge(pred_result["label_raw"], pred_result["label_pt"])
            st.metric("Classificação", pred_result["label_pt"])
            st.metric("IMC", f'{pred_result["bmi"]:.1f}')
            if pred_result["probs_top"]:
                st.markdown('<div class="card"><strong>Top 3 probabilidades</strong></div>', unsafe_allow_html=True)
                for cls, prob in pred_result["probs_top"]:
                    st.write(f"- {cls}: {prob:.1%}")
        else:
            st.markdown(
                """
                <div class="result-panel result-empty">
                    <h3 style="margin-bottom:0.4rem;">Resultado</h3>
                    <p>Preencha o formulário e clique em <strong>"Prever Classificação"</strong><br/>para visualizar os resultados.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

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





