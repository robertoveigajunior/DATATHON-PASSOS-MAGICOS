import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor

# Título do aplicativo
st.set_page_config(
    page_title='Projeto de Análise Preditiva da ONG Passos Mágicos',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# Descrição do projeto
'''
# :earth_americas: Projeto de Análise Preditiva da ONG Passos Mágicos
O projeto visa desenvolver um modelo preditivo para avaliar o impacto da ONG "Passos Mágicos" no desenvolvimento educacional de crianças e jovens em situação de vulnerabilidade. Utilizando dados de 2020 a 2023, o objetivo é identificar estudantes em risco de dificuldades de aprendizado e otimizar a alocação de recursos da ONG. O projeto culmina em um dashboard interativo que permite à equipe da ONG visualizar insights e previsões em tempo real.
'''

MIN_YEAR = 2020
MAX_YEAR = 2022

# Slider para selecionar o ano de interesse
# Seleção do ano de interesse
from_year = st.slider('Selecione o ano de interesse:', min_value=MIN_YEAR, max_value=MAX_YEAR, value=MIN_YEAR)

df = pd.read_csv(r'./PEDE_PASSOS_DATASET_FIAP1.csv', sep=';')

# st.write(df.head())

df.columns[df.columns.str.contains('2020')]
df.columns[df.columns.str.contains('2021')]
df.columns[df.columns.str.contains('2022')]

df[['NOME', 'INDE_2020', 'INDE_2021', 'INDE_2022']].dropna()

# FAZENDO O FILTRO PARA OBTER OS DADOS POR ANO E FAZENDO A LIMPEZA DOS DADOS

def filtered(df, filters):
    def filtrar_colunas(column):
        return not any(map(lambda filter: filter in column, filters))
    index = list(map(filtrar_colunas, df.columns))
    return df.columns[index]

def limpar_dados(df):
  _df = df.dropna(subset=df.columns.difference(['NOME']), how='all')
  _df = _df[~_df.isna().all(axis=1)]
  return _df.reset_index(drop=True)


df_2020 = df[filtered(df, ['2021', '2022'])]
df_2021 = df[filtered(df, ['2020', '2022'])]
df_2022 = df[filtered(df, ['2020', '2021'])]

df_2020 = limpar_dados(df_2020)
df_2021 = limpar_dados(df_2021)
df_2022 = limpar_dados(df_2022)

# CONVERTENDO AS COLUNAS TIPO STRING PARA TIPO NUMÉRICO
# Selecionando as colunas float
float_columns_2020 = ['INDE_2020', 'IAA_2020', 'IEG_2020', 'IPS_2020', 'IDA_2020', 'IPP_2020', 'IPV_2020']
float_columns_2021 = ['INDE_2021', 'IAA_2021', 'IEG_2021', 'IPS_2021', 'IDA_2021', 'IPP_2021', 'IPV_2021']
float_columns_2022 = ['INDE_2022', 'IAA_2022', 'IEG_2022', 'IPS_2022', 'IDA_2022', 'IPP_2022', 'IPV_2022']

# Selecionando as colunas int
int_columns_2020 = ['IDADE_ALUNO_2020', 'ANOS_PM_2020', 'IAN_2020']
int_columns_2021 = ['IAN_2021', 'FASE_2021', 'DEFASAGEM_2021']
int_columns_2022 = ['IAN_2022', 'FASE_2022']

def convert_to_float(x):
    try:
        return round(float(x), 2)
    except:
        return 0

def convert_to_int(x):
    try:
        return int(x)
    except:
        return 0


df_2020[float_columns_2020] = df_2020[float_columns_2020].map(convert_to_float)
df_2021[float_columns_2021] = df_2021[float_columns_2021].map(convert_to_float)
df_2022[float_columns_2022] = df_2022[float_columns_2022].map(convert_to_float)

df_2020[int_columns_2020] = df_2020[int_columns_2020].map(convert_to_int)
df_2021[int_columns_2021] = df_2021[int_columns_2021].map(convert_to_int)
df_2022[int_columns_2022] = df_2022[int_columns_2022].map(convert_to_int)

# Unindos colunas méricas (int e float)
int_columns_2020.extend(float_columns_2020)
int_columns_2021.extend(float_columns_2021)
int_columns_2022.extend(float_columns_2022)

# Atribuindo para uma novas lista
colunas_numericas_2020 = int_columns_2020
colunas_numericas_2021 = int_columns_2021
colunas_numericas_2022 = int_columns_2022

numerical_df = df_2020.select_dtypes(include=['float', 'int'])

correlation_matrix = numerical_df.corr()

fig, ax = plt.subplots(figsize=(8, 6))

# Criar o gráfico de dispersão
if from_year == 2020:
    sns.scatterplot(x='INDE_2020', y='IAA_2020', data=df_2020, ax=ax)
    ax.set_title('Relação entre INDE e IAA em 2020')
elif from_year == 2021:
    sns.scatterplot(x='INDE_2021', y='IAA_2021', data=df_2021, ax=ax)
    ax.set_title('Relação entre INDE e IAA em 2021')
elif from_year == 2022:
    sns.scatterplot(x='INDE_2022', y='IAA_2022', data=df_2022, ax=ax)
    ax.set_title('Relação entre INDE e IAA em 2022')

# Configurar o título e os rótulos dos eixos
ax.set_xlabel('INDE_2020')
ax.set_ylabel('IAA_2020')

# Mostrar o gráfico no Streamlit
st.pyplot(fig)

# Histograma para mostrar a distribuição de uma variável numérica, por exemplo, 'INDE_2020':
# Criar a figura e os eixos
fig, ax = plt.subplots(figsize=(8, 6))

# Plotar o histograma com base no ano selecionado
if from_year == 2020:
    sns.histplot(df_2020['INDE_2020'], kde=True, ax=ax)
    ax.set_title('Distribuição do INDE em 2020')
elif from_year == 2021:
    sns.histplot(df_2021['INDE_2021'], kde=True, ax=ax)
    ax.set_title('Distribuição do INDE em 2021')
elif from_year == 2022:
    sns.histplot(df_2022['INDE_2022'], kde=True, ax=ax)
    ax.set_title('Distribuição do INDE em 2022')

# Configurar os rótulos dos eixos
ax.set_xlabel('INDE')
ax.set_ylabel('Frequência')

# Mostrar o gráfico no Streamlit
st.pyplot(fig)

# Configurar o título e os rótulos dos eixos
ax.set_xlabel('INDE_2020')
ax.set_ylabel('Frequência')

# Mostrar o gráfico no Streamlit
st.pyplot(fig)

# Criar a figura e os eixos
fig, ax = plt.subplots(figsize=(10, 6))

# Criar o gráfico de barras para a variável categórica
if from_year == 2020:
    df['INSTITUICAO_ENSINO_ALUNO_2020'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Distribuição das Instituições de Ensino dos Alunos em 2020')
elif from_year == 2021:
    df['INSTITUICAO_ENSINO_ALUNO_2021'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Distribuição das Instituições de Ensino dos Alunos em 2021')
elif from_year == 2022:
    df['ANO_INGRESSO_2022'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Distribuição das Instituições de Ensino dos Alunos em 2022')

# Configurar o título e os rótulos dos eixos
ax.set_xlabel('Instituição de Ensino')
ax.set_ylabel('Número de Alunos')

# Mostrar o gráfico no Streamlit
st.pyplot(fig)

# Boxplot para mostrar a distribuição de uma variável numérica em relação a uma variável categórica, por exemplo, 'INSTITUICAO_ENSINO_ALUNO_2020':
fig, ax = plt.subplots()
sns.heatmap(correlation_matrix, annot=True, linewidths=0.5, ax=ax)
st.pyplot(fig)

# Modelo de classificação para o ano de 2020
le = LabelEncoder()

y = le.fit_transform(df_2020['INSTITUICAO_ENSINO_ALUNO_2020'])
X = df_2020[colunas_numericas_2020]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
if from_year == 2020:
    st.markdown("## Resultados da Classificação para 2020")
    st.markdown(f"### Acurácia do Modelo: **{accuracy:.2%}**")
    st.markdown("---")

# Modelo de classificação para o ano de 2021

y = le.fit_transform(df_2021['INSTITUICAO_ENSINO_ALUNO_2021'])
X = df_2021[colunas_numericas_2021]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
if from_year == 2021:
    st.markdown("## Resultados da Classificação para 2021")
    st.markdown(f"### Acurácia do Modelo: **{accuracy:.2%}**")
    st.markdown("---")


# Modelo de classificação para o ano de 2021

y = le.fit_transform(df_2022['ANO_INGRESSO_2022'])
X = df_2022[colunas_numericas_2022]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
if from_year == 2022:
    st.markdown("## Resultados da Classificação para 2022")
    st.markdown(f"### Acurácia do Modelo: **{accuracy:.2%}**")
    st.markdown("---")

"""Regressão Linear"""

# Montar a regressao linear para o ano 2020
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

y = le.fit_transform(df_2020['INSTITUICAO_ENSINO_ALUNO_2020'])
X = df_2020[colunas_numericas_2020]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar o modelo de regressão linear
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
linear_predictions = linear_model.predict(X_test)

# Avaliar o desempenho do modelo
from sklearn.metrics import mean_squared_error, r2_score

if from_year == 2020:
    st.markdown("## Regressão Linear 2020")
    st.write("R²:", r2_score(y_test, linear_predictions))
    st.write("RMSE:", mean_squared_error(y_test, linear_predictions, squared=False))
    st.markdown("---")

# Montar a regressao linear para o ano 2020
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

y = le.fit_transform(df_2021['INSTITUICAO_ENSINO_ALUNO_2021'])
X = df_2021[colunas_numericas_2021]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Treinar o modelo de regressão linear
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
linear_predictions = linear_model.predict(X_test)

# Avaliar o desempenho do modelo
from sklearn.metrics import mean_squared_error, r2_score

if from_year == 2021:
    st.markdown("## Regressão Linear 2021")
    st.write("R²:", r2_score(y_test, linear_predictions))
    st.write("RMSE:", mean_squared_error(y_test, linear_predictions, squared=False))
    st.markdown("---")

# Montar a regressao linear para o ano 2020
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

y = le.fit_transform(df_2022['ANO_INGRESSO_2022'])
X = df_2022[colunas_numericas_2022]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Treinar o modelo de regressão linear
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
linear_predictions = linear_model.predict(X_test)

# Avaliar o desempenho do modelo
from sklearn.metrics import mean_squared_error, r2_score

if from_year == 2022:
    st.markdown("## Regressão Linear 2022")
    st.write("R²:", r2_score(y_test, linear_predictions))
    st.write("RMSE:", mean_squared_error(y_test, linear_predictions, squared=False))
    st.markdown("---")

"""Modelo de Arvore de decisão"""

# Montar modelo de arvore de decisão
from sklearn.tree import DecisionTreeRegressor

y = le.fit_transform(df_2020['INSTITUICAO_ENSINO_ALUNO_2020'])
X = df_2020[colunas_numericas_2020]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar o modelo de árvore de decisão
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
tree_predictions = tree_model.predict(X_test)

if from_year == 2020:
    st.markdown("## Árvore de Decisão 2020")
    st.write("R²:", r2_score(y_test, tree_predictions))
    st.write("RMSE:", mean_squared_error(y_test, tree_predictions, squared=False))
    st.markdown("---")

# Montar modelo de arvore de decisão
from sklearn.tree import DecisionTreeRegressor

y = le.fit_transform(df_2021['INSTITUICAO_ENSINO_ALUNO_2021'])
X = df_2021[colunas_numericas_2021]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar o modelo de árvore de decisão
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
tree_predictions = tree_model.predict(X_test)

if from_year == 2021:
    st.markdown("## Árvore de Decisão 2021")
    st.write("R²:", r2_score(y_test, tree_predictions))
    st.write("RMSE:", mean_squared_error(y_test, tree_predictions, squared=False))
    st.markdown("---")

# Montar modelo de arvore de decisão
from sklearn.tree import DecisionTreeRegressor

y = le.fit_transform(df_2022['ANO_INGRESSO_2022'])
X = df_2022[colunas_numericas_2022]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar o modelo de árvore de decisão
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
tree_predictions = tree_model.predict(X_test)

if from_year == 2022:
    st.markdown("## Árvore de Decisão 2022")
    st.write("R²:", r2_score(y_test, tree_predictions))
    st.write("RMSE:", mean_squared_error(y_test, tree_predictions, squared=False))
    st.markdown("---")

"""Rede Neural"""

y = le.fit_transform(df_2020['INSTITUICAO_ENSINO_ALUNO_2020'])
X = df_2020[colunas_numericas_2020]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar o modelo de rede neural
regressor_model = MLPRegressor(random_state=42)
regressor_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
regressor_predictions = regressor_model.predict(X_test)

if from_year == 2020:
    st.markdown("## Rede Neural 2020")
    st.write("R²:", r2_score(y_test, regressor_predictions))
    st.write("RMSE:", mean_squared_error(y_test, regressor_predictions, squared=False))
    st.markdown("---")

y = le.fit_transform(df_2021['INSTITUICAO_ENSINO_ALUNO_2021'])
X = df_2021[colunas_numericas_2021]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar o modelo de rede neural
regressor_model = MLPRegressor(random_state=42)
regressor_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
regressor_predictions = regressor_model.predict(X_test)

if from_year == 2021:
    st.markdown("## Rede Neural 2021")
    st.write("R²:", r2_score(y_test, regressor_predictions))
    st.write("RMSE:", mean_squared_error(y_test, regressor_predictions, squared=False))
    st.markdown("---")

y = le.fit_transform(df_2022['ANO_INGRESSO_2022'])
X = df_2022[colunas_numericas_2022]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar o modelo de rede neural
regressor_model = MLPRegressor(random_state=42)
regressor_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
regressor_predictions = regressor_model.predict(X_test)

if from_year == 2022:
    st.markdown("## Rede Neural 2022")
    st.write("R²:", r2_score(y_test, regressor_predictions))
    st.write("RMSE:", mean_squared_error(y_test, regressor_predictions, squared=False))
    st.markdown("---")


# Título da seção
st.markdown("## Apuração dos Modelos")
st.markdown("---")

# Dados dos resultados dos modelos
resultados = {
    'Classificação': {
        '2020': {'acuracia': 0.84},
        '2021': {'acuracia': 0.86},
        '2022': {'acuracia': 0.34}
    },
    'Regressão Linear': {
        '2020': {'R²': 0.209, 'RMSE': 1.067},
        '2021': {'R²': 0.276, 'RMSE': 0.315},
        '2022': {'R²': 0.168, 'RMSE': 1.590}
    },
    'Árvore de Decisão': {
        '2020': {'R²': 0.052, 'RMSE': 1.168},
        '2021': {'R²': -0.197, 'RMSE': 0.406},
        '2022': {'R²': -0.463, 'RMSE': 2.109}
    },
    'Rede Neural': {
        '2020': {'R²': 0.353, 'RMSE': 0.965},
        '2021': {'R²': 0.301, 'RMSE': 0.310},
        '2022': {'R²': 0.030, 'RMSE': 1.717}
    }
}

# Melhor modelo de classificação
melhor_classificacao = max(resultados['Classificação'].items(), key=lambda x: x[1]['acuracia'])
st.markdown("### Melhor Modelo de Classificação")
st.markdown(f"- **Ano:** {melhor_classificacao[0]}")
st.markdown(f"- **Acurácia:** {melhor_classificacao[1]['acuracia']:.2%}")
st.markdown("A acurácia representa a proporção de previsões corretas feitas pelo modelo.")

# Melhor modelo de regressão para cada ano
st.markdown("### Melhor Modelo de Regressão por Ano")
for ano in ['2020', '2021', '2022']:
    melhor_regressao = max(
        [(modelo, metricas[ano]) for modelo, metricas in resultados.items() if modelo != 'Classificação'],
        key=lambda x: x[1]['R²']
    )
    st.markdown(f"#### Ano {ano}")
    st.markdown(f"- **Modelo:** {melhor_regressao[0]}")
    st.markdown(f"- **R²:** {melhor_regressao[1]['R²']:.3f}")
    st.markdown(f"- **RMSE:** {melhor_regressao[1]['RMSE']:.3f}")
    st.markdown("O R² indica a proporção da variabilidade dos dados explicada pelo modelo, enquanto o RMSE mede o erro médio das previsões.")

st.markdown("---")

"""**Conclusão Classificação:**

1. **Melhor Modelo de Regressão por Ano:**
* 2020: O modelo de Rede Neural foi identificado como o melhor, com um R² de 0.353, indicando que ele explica 35.3% da variabilidade dos dados, e um RMSE de 0.965, sugerindo um erro médio nas previsões.
* 2021: Novamente, a Rede Neural se destacou, com um R² de 0.301 e um RMSE de 0.31, mostrando uma performance consistente em relação ao ano anterior.
* 2022: A Regressão Linear foi o melhor modelo, com um R² de 0.168, explicando 16.8% da variabilidade, e um RMSE de 1.59, indicando um maior erro médio comparado aos anos anteriores.

2. **Melhor Modelo de Classificação:**
* Para o ano de 2021, o melhor modelo de classificação alcançou uma acurácia de 0.86, o que significa que ele classificou corretamente 86% dos casos, demonstrando uma boa capacidade de previsão para esse ano específico.



Conclusão:
A análise revela que, embora a Rede Neural tenha sido o melhor modelo de regressão para 2020 e 2021, sua capacidade de explicação dos dados ainda é limitada, como indicado pelos valores de R². Em 2022, a Regressão Linear foi mais adequada, apesar de também apresentar um R² relativamente baixo. O modelo de classificação para 2021, no entanto, mostrou uma alta acurácia, destacando-se como uma ferramenta eficaz para categorizar os dados desse ano.
"""

