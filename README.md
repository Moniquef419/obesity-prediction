# Previsor de Obesidade â€” Hospital

Modelo preditivo de Machine Learning para auxiliar mÃ©dicos a diagnosticar nÃ­veis de obesidade em pacientes.

## Requisitos

- **Acuracia**: 96% (bem acima do minimo de 75%)
- **Modelo**: Random Forest com GridSearchCV
- **Framework**: Streamlit para interface web
- **Dataset**: 2.111 amostras com 16 features + 1 target

## Estrutura do Projeto

```
.
â”œâ”€â”€ app.py                      # App principal Streamlit (prediÃ§Ã£o + painel)
â”œâ”€â”€ requirements.txt            # DependÃªncias Python
â”œâ”€â”€ README.md                   # Este arquivo
â”œâ”€â”€ Obesity.csv                 # Dataset de treino/teste
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ __init__.py             # Package marker
â”‚   â”œâ”€â”€ eda.py                  # ExploraÃ§Ã£o de dados
â”‚   â”œâ”€â”€ data_prep.py            # Preprocessamento e feature engineering
â”‚   â””â”€â”€ train.py                # Treinamento do modelo
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ obesity_pipeline.joblib # Pipeline treinado (Random Forest + Preprocessor)
â”‚   â””â”€â”€ label_encoder.joblib    # Label encoder do target (7 classes)
â””â”€â”€ .streamlit/
    â””â”€â”€ config.toml             # ConfiguraÃ§Ãµes do Streamlit
```

## Features Utilizadas

- **DemogrÃ¡ficas**: Gender, Age, Height, Weight, BMI (calculado)
- **Comportamentais**: FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS
- **MÃ©dicas**: family_history

## Classes do Target (Obesity)

- 0: Insufficient_Weight
- 1: Normal_Weight
- 2: Overweight_Level_I
- 3: Overweight_Level_II
- 4: Obesity_Type_I
- 5: Obesity_Type_II
- 6: Obesity_Type_III

## Como Usar Localmente

1. **Clone ou copie o repositÃ³rio**:
   ```bash
   cd seu_projeto
   ```

2. **Instale as dependÃªncias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Treinar o modelo (opcional, jÃ¡ estÃ¡ em `models/`):**
   ```bash
   python src/train.py
   ```

4. **Rodando a aplicaÃ§Ã£o Streamlit:**
   ```bash
   streamlit run app.py
   ```
   Abra `http://localhost:8501` no seu navegador.

## Abas da AplicaÃ§Ã£o

### 1. **Prever**
- FormulÃ¡rio para entrada de dados do paciente
- Retorna a classe predita e probabilidades
- Ideal para suporte Ã  decisÃ£o clÃ­nica

### 2. **Painel AnalÃ­tico**
- DistribuiÃ§Ã£o das classes de obesidade
- Boxplot BMI por categoria
- Scatter plot Idade vs BMI (colorido por classe)
- Amostra dos dados

## MÃ©tricas de Desempenho

```
Accuracy: 96%
Macro Avg F1: 0.96
Weighted Avg F1: 0.96
```

**Detalhes por classe:**
- Insufficient_Weight (0): F1 = 0.99
- Normal_Weight (1): F1 = 0.90
- Overweight_Level_I (2): F1 = 0.99
- Overweight_Level_II (3): F1 = 0.98
- Obesity_Type_I (4): F1 = 0.99
- Obesity_Type_II (5): F1 = 0.91
- Obesity_Type_III (6): F1 = 0.99

## Deploy no Streamlit Cloud

1. FaÃ§a push do repositÃ³rio para GitHub
2. Acesse [Streamlit Cloud](https://streamlit.io/cloud)
3. Clique "New app"
4. Selecione o repositÃ³rio e arquivo `app.py`
5. Deploy serÃ¡ automÃ¡tico

## DependÃªncias Principais

- `pandas`, `numpy`: ManipulaÃ§Ã£o de dados
- `scikit-learn`: Treinamento e preprocessamento
- `streamlit`: Interface web
- `matplotlib`, `seaborn`: VisualizaÃ§Ãµes
- `joblib`: SerializaÃ§Ã£o do modelo

## Notas

- O modelo usa um Pipeline do sklearn que jÃ¡ inclui preprocessing (StandardScaler para numÃ©ricos, OneHotEncoder para categÃ³ricos)
- O target Ã© codificado (0-6) durante treinamento e decodificado na saÃ­da
- Todos os pacientes devem fornecer todas as 15 features para prediÃ§Ã£o

## Autor

Desenvolvido como soluÃ§Ã£o ao Tech Challenge da FIAP â€” PrevisÃ£o de Obesidade

---

**VersÃ£o**: 1.0  
**Data**: Fevereiro/2026


