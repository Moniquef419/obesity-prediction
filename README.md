# Previsor de Obesidade - Hospital

Aplicacao de Machine Learning para apoiar a equipe medica na classificacao do nivel de obesidade de pacientes.

## Objetivo

Desenvolver um pipeline completo de ML com deploy em Streamlit para:

- prever a classe de obesidade de um paciente
- oferecer um painel analitico com insights dos dados
- atender ao requisito de desempenho minimo (acuracia > 75%)

## Resultado

- Acuracia de teste: **96%**
- Macro F1: **0.96**
- Weighted F1: **0.96**

## Estrutura do Projeto

```text
.
|-- app.py
|-- requirements.txt
|-- README.md
|-- ENTREGA.txt
|-- Obesity.csv
|-- data_sample.csv
|-- models/
|   |-- obesity_pipeline.joblib
|   `-- label_encoder.joblib
`-- src/
    |-- __init__.py
    |-- eda.py
    |-- data_prep.py
    `-- train.py
```

## Pipeline de ML

1. Leitura de dados (`Obesity.csv`)
2. Feature engineering com:
- calculo de `BMI`
- tratamento de variaveis binarias (`yes/no`)
- padronizacao de variaveis de escala por arredondamento (`FCVC`, `NCP`, `CH2O`, `FAF`, `TUE`)
3. Preprocessamento com `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`)
4. Treinamento com `RandomForestClassifier` + `GridSearchCV`
5. Serializacao com `joblib`

## Como Executar Localmente

1. Instale dependencias:

```bash
pip install -r requirements.txt
```

2. (Opcional) Re-treine o modelo:

```bash
python src/train.py
```

3. Rode a aplicacao:

```bash
streamlit run app.py
```

4. Abra no navegador:

```text
http://localhost:8501
```

## Funcionalidades da Aplicacao

### 1) Prever

- formulario em portugues
- conversao interna para os codigos esperados pelo modelo
- resultado da classe prevista
- top 3 probabilidades

### 2) Painel Analitico

- filtros por genero, idade e classe
- metricas chave (BMI medio, risco, etc.)
- distribuicoes e comparativos para suporte clinico

## Classes de Saida (Target)

- `Insufficient_Weight`
- `Normal_Weight`
- `Overweight_Level_I`
- `Overweight_Level_II`
- `Obesity_Type_I`
- `Obesity_Type_II`
- `Obesity_Type_III`

## Deploy no Streamlit Cloud

1. Suba o projeto no GitHub
2. No Streamlit Cloud, crie um app apontando para:
- repositorio: `Moniquef419/obesity-prediction`
- branch: `main`
- arquivo principal: `app.py`
3. Clique em deploy/reboot para atualizar

## Principais Tecnologias

- Python
- pandas, numpy
- scikit-learn
- streamlit
- matplotlib, seaborn
- joblib

## Links

- Repositorio: https://github.com/Moniquef419/obesity-prediction
- Aplicacao: https://obesity-prediction-yf9mmuflqmawja7quzv29x.streamlit.app/

## Versao

- Versao: 1.0
- Data: Fevereiro/2026
