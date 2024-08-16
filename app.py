import streamlit as st
import pandas as pd
import requests
import numpy as np

# Título do aplicativo
st.title('DATATHON PASSOS MÁGICOS')

# Descrição do projeto
st.write("""
O projeto visa desenvolver um modelo preditivo para avaliar o impacto da ONG "Passos Mágicos" no desenvolvimento educacional de crianças e jovens em situação de vulnerabilidade. Utilizando dados de 2020 a 2023, o objetivo é identificar estudantes em risco de dificuldades de aprendizado e otimizar a alocação de recursos da ONG. O projeto culmina em um dashboard interativo que permite à equipe da ONG visualizar insights e previsões em tempo real.
""")

import pandas as pd

df = pd.read_csv("PEDE_PASSOS_DATASET_FIAP.csv", sep=";")

df = df.dropna()

df = df.drop_duplicates()

print(df.head())

print(df.info())

print(df.head())

print(df.describe(include='all'))

# df.hist(figsize=(15, 10))
# plt.show()

# sns.boxplot(data=df)
# plt.show()

# Gráficos de barras para variáveis categóricas (exemplo)
# df['INSTITUICAO_ENSINO_ALUNO_2020'].value_counts().plot(kind='bar')
# plt.show()

# Gráficos de dispersão (exemplo)
# plt.scatter(df['IDADE_ALUNO_2020'], df['INDE_2020'])
# plt.xlabel('Idade do Aluno em 2020')
# plt.ylabel('IDADE 2020')
# plt.show()

# 3. Análise de Correlação
# Matriz de correlação
# correlation_matrix = df.corr()
# print(correlation_matrix)
# Select only numerical columns
numerical_df = df.select_dtypes(include=['float', 'int'])

# Calculate the correlation matrix
correlation_matrix = numerical_df.corr()

# Print the correlation matrix
print(correlation_matrix)

# Heatmap da matriz de correlação
# sns.heatmap(correlation_matrix, annot=True)
# plt.show()

# 4. Feature Engineering (Exemplo)
# Criar variável combinada: Renda per capita familiar
# (assumindo que exista uma coluna 'RENDA_FAMILIAR' e 'NUM_PESSOAS_RESIDENCIA')
# df['RENDA_PER_CAPITA'] = df['RENDA_FAMILIAR'] / df['NUM_PESSOAS_RESIDENCIA']

# Converter variáveis categóricas em dummies (exemplo)
# df = pd.get_dummies(df, columns=['INSTITUICAO_ENSINO_ALUNO_2020'])

# Imprimir as primeiras linhas do DataFrame modificado (após Feature Engineering)
print(df.head())

# Divisão e treinamento dos dados

X = df.drop('INDE_2020', axis=1)  # Substitua 'INDE_2020' pela coluna alvo real
y = df['INDE_2020']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Assuming 'INSTITUICAO_ENSINO_ALUNO_2020' is the column with the string
# X_train = pd.get_dummies(X_train, columns=['INSTITUICAO_ENSINO_ALUNO_2020'], drop_first=True) # Correct column name
# X_test = pd.get_dummies(X_test, columns=['INSTITUICAO_ENSINO_ALUNO_2020'], drop_first=True) # Correct column name

# Modelos

# Verifique se há colunas não numéricas restantes em X_train e X_test
# print(X_train.select_dtypes(include=['object']).columns)
# print(X_test.select_dtypes(include=['object']).columns) 

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()

# Iterar sobre colunas não numéricas e aplicar LabelEncoder
# for col in X_train.select_dtypes(include=['object']).columns:
    # Ajuste o LabelEncoder nos valores exclusivos combinados do treinamento e do teste
    # all_values = list(X_train[col].unique()) + list(X_test[col].unique())
    # le.fit(all_values)  
    # X_train[col] = le.transform(X_train[col])
    # Use o mesmo codificador instalado nos dados de treinamento para o conjunto de teste
    # X_test[col] = le.transform(X_test[col])  

# Obter colunas ausentes em X_test
# missing_cols = set(X_train.columns) - set(X_test.columns)
# Add a missing column in X_test
# for c in missing_cols:
    # X_test[c] = 0

# Garanta que a ordem das colunas no conjunto de teste esteja na mesma ordem que no conjunto de treinamento
# X_test = X_test[X_train.columns]

# linear_model = LinearRegression()
# linear_model.fit(X_train, y_train)
# linear_predictions = linear_model.predict(X_test)

# Avaliação do modelo Arvore de Decisão
# tree_model = DecisionTreeRegressor(random_state=42)
# tree_model.fit(X_train, y_train)
# tree_predictions = tree_model.predict(X_test)

# Avaliação do modelo Rede Neural
# neural_model = MLPRegressor(random_state=42, max_iter=500)  # Aqui podemos ajustar o max_iter se necessário
# neural_model.fit(X_train, y_train)
# neural_predictions = neural_model.predict(X_test)

# Avaliação dos modelos
print("----- Regressão Linear -----")
# print("R²:", r2_score(y_test, linear_predictions))
# print("RMSE:", mean_squared_error(y_test, linear_predictions, squared=False))

print("\n----- Árvore de Decisão -----")
# print("R²:", r2_score(y_test, tree_predictions))
# print("RMSE:", mean_squared_error(y_test, tree_predictions, squared=False))

print("\n----- Rede Neural -----")
# print("R²:", r2_score(y_test, neural_predictions))
# print("RMSE:", mean_squared_error(y_test, neural_predictions, squared=False))
