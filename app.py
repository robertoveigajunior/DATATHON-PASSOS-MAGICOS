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

# Remover valores ausentes
df = df.dropna()

# Remover duplicatas
df = df.drop_duplicates()
st.write("Primeiras linhas do DataFrame:")
st.write(df.head())

# Análise Descritiva
st.write("Estatísticas descritivas:")
st.write(df.describe(include='all'))

df['INSTITUICAO_ENSINO_ALUNO_2020'].value_counts().plot(kind='bar')
st.write("Gráfico de barras para variáveis categóricas:")
st.write(df.info())

# informações sobre o DataFrame
# st.write("Informações sobre o DataFrame:")
# st.write(df.info())

# # Dividir o DataFrame em variáveis independentes (X) e dependente (y)
# X = df.drop(columns='INDE_2020')
# y = df['IDADE_2020']

# # Dividir o DataFrame em treino e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Treinar um modelo de Regressão Linear
# linear_model = LinearRegression()
# linear_model.fit(X_train, y_train)

# # Fazer previsões
# y_pred = linear_model.predict(X_test)

# # Avaliar o modelo
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# st.write("Métricas do modelo de Regressão Linear:")
# st.write(f"MSE: {mse}")
# st.write(f"R2: {r2}")

# # Treinar um modelo de Árvore de Decisão
# tree_model = DecisionTreeRegressor()
# tree_model.fit(X_train, y_train)

# # Fazer previsões
# y_pred = tree_model.predict(X_test)

# # Avaliar o modelo
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# st.write("Métricas do modelo de Árvore de Decisão:")
# st.write(f"MSE: {mse}")
# st.write(f"R2: {r2}")

# # Treinar um modelo de Rede Neural
# nn_model = MLPRegressor()
# nn_model.fit(X_train, y_train)

# # Fazer previsões
# y_pred = nn_model.predict(X_test)

# # Avaliar o modelo
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# st.write("Métricas do modelo de Rede Neural:")
# st.write(f"MSE: {mse}")
# st.write(f"R2: {r2}")

# # Análise Exploratória de Dados
# st.write("Análise Exploratória de Dados:")
# # Contagem de valores para variáveis categóricas
# st.write("Contagem de valores para variáveis categóricas:")
# st.write(df['INSTITUICAO_ENSINO_ALUNO_2020'].value_counts())

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
# fig, ax = plt.subplots()
# ax.scatter(df['IDADE_ALUNO_2020'], df['INDE_2020'])
# ax.set_xlabel('Idade do Aluno em 2020')
# ax.set_ylabel('IDADE 2020')
# st.pyplot(fig)

# Análise de Correlação
st.write("Matriz de correlação:")
numerical_df = df.select_dtypes(include=['float', 'int'])
correlation_matrix = numerical_df.corr()
st.write(correlation_matrix)

# Heatmap da matriz de correlação
st.write("Heatmap da matriz de correlação:")
# fig, ax = plt.subplots()
# sns.heatmap(correlation_matrix, annot=True, ax=ax)
# st.pyplot(fig)

# Feature Engineering (Exemplo)
# Criar variável combinada: Renda per capita familiar
# (assumindo que exista uma coluna 'RENDA_FAMILIAR' e 'NUM_PESSOAS_RESIDENCIA')
# df['RENDA_PER_CAPITA'] = df['RENDA_FAMILIAR'] / df['NUM_PESSOAS_RESIDENCIA']

# Converter variáveis categóricas em dummies (exemplo)
# df = pd.get_dummies(df, columns=['INSTITUICAO_ENSINO_ALUNO_2020'])

# Imprimir as primeiras linhas do DataFrame modificado (após Feature Engineering)
st.write("Primeiras linhas do DataFrame modificado:")
st.write(df.head())