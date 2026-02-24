# Previsor de Obesidade — Hospital

Modelo preditivo de Machine Learning para auxiliar médicos a diagnosticar níveis de obesidade em pacientes.

## Requisitos

- **Acurácia**: 97% (bem acima do mínimo de 75%)
- **Modelo**: Random Forest com GridSearchCV
- **Framework**: Streamlit para interface web
- **Dataset**: 2.111 amostras com 16 features + 1 target

## Estrutura do Projeto

```
.
├── app.py                      # App principal Streamlit (predição + painel)
├── requirements.txt            # Dependências Python
├── README.md                   # Este arquivo
├── Obesity.csv                 # Dataset de treino/teste
├── src/
│   ├── __init__.py             # Package marker
│   ├── eda.py                  # Exploração de dados
│   ├── data_prep.py            # Preprocessamento e feature engineering
│   └── train.py                # Treinamento do modelo
├── models/
│   ├── obesity_pipeline.joblib # Pipeline treinado (Random Forest + Preprocessor)
│   └── label_encoder.joblib    # Label encoder do target (7 classes)
└── .streamlit/
    └── config.toml             # Configurações do Streamlit
```

## Features Utilizadas

- **Demográficas**: Gender, Age, Height, Weight, BMI (calculado)
- **Comportamentais**: FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS
- **Médicas**: family_history

## Classes do Target (Obesity)

- 0: Insufficient_Weight
- 1: Normal_Weight
- 2: Overweight_Level_I
- 3: Overweight_Level_II
- 4: Obesity_Type_I
- 5: Obesity_Type_II
- 6: Obesity_Type_III

## Como Usar Localmente

1. **Clone ou copie o repositório**:
   ```bash
   cd seu_projeto
   ```

2. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Treinar o modelo (opcional, já está em `models/`):**
   ```bash
   python src/train.py
   ```

4. **Rodando a aplicação Streamlit:**
   ```bash
   streamlit run app.py
   ```
   Abra `http://localhost:8501` no seu navegador.

## Abas da Aplicação

### 1. **Prever**
- Formulário para entrada de dados do paciente
- Retorna a classe predita e probabilidades
- Ideal para suporte à decisão clínica

### 2. **Painel Analítico**
- Distribuição das classes de obesidade
- Boxplot BMI por categoria
- Scatter plot Idade vs BMI (colorido por classe)
- Amostra dos dados

## Métricas de Desempenho

```
Accuracy: 97%
Macro Avg F1: 0.97
Weighted Avg F1: 0.97
```

**Detalhes por classe:**
- Insufficient_Weight (0): F1 = 0.99
- Normal_Weight (1): F1 = 0.92
- Overweight_Level_I (2): F1 = 0.99
- Overweight_Level_II (3): F1 = 0.99
- Obesity_Type_I (4): F1 = 0.99
- Obesity_Type_II (5): F1 = 0.94
- Obesity_Type_III (6): F1 = 0.99

## Deploy no Streamlit Cloud

1. Faça push do repositório para GitHub
2. Acesse [Streamlit Cloud](https://streamlit.io/cloud)
3. Clique "New app"
4. Selecione o repositório e arquivo `app.py`
5. Deploy será automático

## Dependências Principais

- `pandas`, `numpy`: Manipulação de dados
- `scikit-learn`: Treinamento e preprocessamento
- `streamlit`: Interface web
- `matplotlib`, `seaborn`: Visualizações
- `joblib`: Serialização do modelo

## Notas

- O modelo usa um Pipeline do sklearn que já inclui preprocessing (StandardScaler para numéricos, OneHotEncoder para categóricos)
- O target é codificado (0-6) durante treinamento e decodificado na saída
- Todos os pacientes devem fornecer todas as 15 features para predição

## Autor

Desenvolvido como solução ao Tech Challenge da FIAP — Previsão de Obesidade

---

**Versão**: 1.0  
**Data**: Fevereiro/2026
