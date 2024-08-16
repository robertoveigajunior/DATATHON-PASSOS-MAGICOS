# Projeto de Análise Preditiva da ONG "Passos Mágicos"

## Visão Geral

O projeto visa desenvolver um modelo preditivo para avaliar o impacto da ONG "Passos Mágicos" no desenvolvimento educacional de crianças e jovens em situação de vulnerabilidade. Utilizando dados de 2020 a 2023, o objetivo é identificar estudantes em risco de dificuldades de aprendizado e otimizar a alocação de recursos da ONG. O projeto culmina em um dashboard interativo que permite à equipe da ONG visualizar insights e previsões em tempo real.

## Tecnologias Utilizadas

- **Linguagem de Programação:** Python
- **Bibliotecas de Machine Learning:** Scikit-learn para modelagem preditiva
- **Bibliotecas de Visualização:** Matplotlib e Seaborn para gráficos e visualizações
- **Framework de Dashboard:** Streamlit para criar interfaces interativas
- **Plataforma de Deploy:** Heroku para hospedar e disponibilizar o dashboard online

## Requisitos

- **Python 3.7+**
- **Bibliotecas Python:** pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit
- **Conta no Heroku:** Para deploy do dashboard
- **Git:** Para controle de versão e deploy no Heroku

## Instalação

1. Clone o repositório do projeto:
bash
git clone https://github.com/robertoveigajunior/DATATHON-PASSOS-MAGICOS.git

cd DATATHON-PASSOS-MAGICOS


2. Crie um ambiente virtual e ative-o:
bash
python -m venv venv
source venv/bin/activate  # No Windows use venv\Scripts\activate


3. Instale as dependências:
bash
pip install -r requirements.txt


4. Execute o dashboard localmente:
bash
streamlit run dashboard.py


## Deploy no Heroku

1. Faça login no Heroku:
bash
heroku login


2. Crie um novo aplicativo no Heroku:
bash
heroku create datathon-passos-magicos


3. Configure o Git para o Heroku:
bash
git remote add heroku [https://git.heroku.com/datathon-passos-magicos.git](https://dashboard.heroku.com/apps/app-datathon-passos-magicos)


4. Faça o deploy do aplicativo:
bash
git push heroku main


5. Acesse o aplicativo no navegador:
bash
heroku open

## Grafico

![image](https://github.com/user-attachments/assets/1b8213e9-9bc2-4709-859a-a8223af284fb)


## Conclusão

Link do Dashboard (Sreamlit): https://datathon-paapps-magicos-gz9u7ueehqyzgneyyviwea.streamlit.app/

# Regressão Linear:
R²: 0.5846

RMSE: 0.5262

A Regressão Linear apresenta um R² de 0.5846, indicando que cerca de 58.46% da variabilidade dos dados é explicada pelo modelo. O RMSE de 0.5262 sugere um erro médio razoável nas previsões.

# Árvore de Decisão:
R²: 0.7618

RMSE: 0.3985

A Árvore de Decisão tem um R² de 0.7618, o que significa que ela explica 76.18% da variabilidade dos dados, superando a Regressão Linear. O RMSE de 0.3985 é o menor entre os modelos, indicando que a Árvore de Decisão tem o menor erro médio nas previsões.

# Rede Neural:
R²: -0.8513

RMSE: 1.1108

A Rede Neural apresenta um R² negativo (-0.8513), o que indica que o modelo não está capturando a variabilidade dos dados de forma eficaz e está performando pior do que um modelo que simplesmente prevê a média dos dados. O RMSE de 1.1108 é o maior, sugerindo um erro médio elevado nas previsões.

# Conclusão:

Árvore de Decisão é o modelo com melhor performance, apresentando o maior R² e o menor RMSE, o que indica que ele é mais eficaz em capturar a variabilidade dos dados e em fazer previsões precisas.
Regressão Linear tem uma performance razoável, mas inferior à Árvore de Decisão.
Rede Neural não está performando bem neste caso, possivelmente devido a um ajuste inadequado dos hiperparâmetros ou à necessidade de mais dados ou pré-processamento.


## Storytelling do Projeto

O projeto começou com a missão de entender e amplificar o impacto da ONG "Passos Mágicos" na vida de jovens em vulnerabilidade. Através de uma análise detalhada dos dados educacionais, desenvolvemos um modelo preditivo que não só identifica estudantes em risco, mas também fornece insights valiosos para intervenções personalizadas. Com a implementação de um dashboard interativo, a equipe da ONG agora pode monitorar e ajustar suas estratégias em tempo real, garantindo que cada criança receba o suporte necessário para prosperar. Este projeto não é apenas uma análise de dados, mas uma ferramenta poderosa para transformação social.
