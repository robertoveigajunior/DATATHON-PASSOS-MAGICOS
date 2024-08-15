# DATATHON-PASSOS-MAGICOS

## Descrição do projeto

O projeto DATATHON-PASSOS-MAGICOS é uma competição de ciência de dados que tem como objetivo resolver um problema específico utilizando técnicas de aprendizado de máquina. Nosso foco foi na utilização de.

Nosso objetivo foi criar um modelo preditivo para identificar e prever o desenvolvimento educacional dos estudantes atendidos pela ONG, permitindo intervenções mais eficazes e personalizadas.

Exploramos três modelos de Machine Learning: Regressão Linear, Árvore de Decisão e Rede Neural, para identificar o mais eficaz.

Árvore de Decisão como algoritmo principal para abordar o problema proposto. No entanto, também é importante considerar ajustes nos hiperparâmetros da.

Rede Neural ou explorar outras arquiteturas de modelos de aprendizado de máquina para potencialmente melhorar a performance do sistema. 


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
cd seu-projeto


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
git remote add heroku https://git.heroku.com/datathon-passos-magicos.git


4. Faça o deploy do aplicativo:
bash
git push heroku main


5. Acesse o aplicativo no navegador:
bash
heroku open


## Storytelling do Projeto

O projeto começou com a missão de entender e amplificar o impacto da ONG "Passos Mágicos" na vida de jovens em vulnerabilidade. Através de uma análise detalhada dos dados educacionais, desenvolvemos um modelo preditivo que não só identifica estudantes em risco, mas também fornece insights valiosos para intervenções personalizadas. Com a implementação de um dashboard interativo, a equipe da ONG agora pode monitorar e ajustar suas estratégias em tempo real, garantindo que cada criança receba o suporte necessário para prosperar. Este projeto não é apenas uma análise de dados, mas uma ferramenta poderosa para transformação social.