import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("PEDE_PASSOS_DATASET_FIAP.csv", sep=";")
raw_gdp_df = pd.read_csv(df)

MIN_YEAR = 2020
MAX_YEAR = 2025

gdp_df = raw_gdp_df.melt(
        ['Country Code'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'GDP',
    )

# Remover valores ausentes
df = df.dropna()

# Remover duplicatas
df = df.drop_duplicates()
st.write("Primeiras linhas do DataFrame:")
st.write(df.head())

# Add some spacing
''
''

min_value = gdp_df['Year'].min()
max_value = gdp_df['Year'].max()

from_year, to_year = st.slider(
    'Which years are you interested in?',
    min_value=min_value,
    max_value=max_value,
    value=[min_value, max_value])

countries = gdp_df['Country Code'].unique()

if not len(countries):
    st.warning("Select at least one country")

selected_countries = st.multiselect(
    'Which countries would you like to view?',
    countries,
    ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN'])

''
''
''


# Análise Descritiva
st.write("Estatísticas descritivas:")
st.write(df.describe(include='all'))

df['INSTITUICAO_ENSINO_ALUNO_2020'].value_counts().plot(kind='bar')
st.write("Gráfico de barras para variáveis categóricas:")
st.write(df['INSTITUICAO_ENSINO_ALUNO_2020'].value_counts())

# Histogramas para variáveis numéricas
st.write("Histogramas para variáveis numéricas:")
fig, ax = plt.subplots(figsize=(15, 10))
df.hist(ax=ax)
st.pyplot(fig)

# Boxplots para variáveis numéricas
st.write("Boxplots para variáveis numéricas:")
fig, ax = plt.subplots()
sns.boxplot(data=df, ax=ax)
st.pyplot(fig)

# Gráficos de barras para variáveis categóricas
st.write("Gráficos de barras para variáveis categóricas:")
fig, ax = plt.subplots()
df['INSTITUICAO_ENSINO_ALUNO_2020'].value_counts().plot(kind='bar', ax=ax)
st.pyplot(fig)

# Gráficos de dispersão
st.write("Gráficos de dispersão:")
fig, ax = plt.subplots()
ax.scatter(df['IDADE_ALUNO_2020'], df['INDE_2020'])
ax.set_xlabel('Idade do Aluno em 2020')
ax.set_ylabel('IDADE 2020')
st.pyplot(fig)

# Análise de Correlação
st.write("Matriz de correlação:")
numerical_df = df.select_dtypes(include=['float', 'int'])
correlation_matrix = numerical_df.corr()
st.write(correlation_matrix)

# Heatmap da matriz de correlação
st.write("Heatmap da matriz de correlação:")
fig, ax = plt.subplots()
sns.heatmap(correlation_matrix, annot=True, ax=ax)
st.pyplot(fig)

# Converter variáveis categóricas em dummies (exemplo)
df = pd.get_dummies(df, columns=['INSTITUICAO_ENSINO_ALUNO_2020'])

# Conclusão da Análise Exploratória
st.write("Conclusão da Análise Exploratória:")
st.write("A análise exploratória é uma etapa fundamental em um projeto de ciência de dados. Ela permite entender melhor o conjunto de dados e identificar padrões e tendências que podem ser úteis para a construção de modelos preditivos.")
st.write("Neste projeto, realizamos uma análise exploratória simples de um conjunto de dados de alunos. Foram apresentadas diversas visualizações, como histogramas, boxplots, gráficos de barras e gráficos de dispersão.")
st.write("Também calculamos estatísticas descritivas e analisamos a matriz de correlação entre as variáveis numéricas.")
st.write("Por fim, convertemos variáveis categóricas em dummies para facilitar a construção de modelos preditivos.")
st.write("Essa análise exploratória é apenas um ponto de partida e pode ser aprofundada com a utilização de técnicas estatísticas mais avançadas e a construção de modelos preditivos.")

# Construção de Modelos Preditivos
st.write("Construção de Modelos Preditivos:")
st.write("Nesta seção, iremos construir modelos preditivos para prever a variável alvo 'INDE_2020'.")