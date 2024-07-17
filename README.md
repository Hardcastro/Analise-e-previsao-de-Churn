# Previsão de Churn de Clientes
Este repositório contém um projeto que prevê o churn de clientes usando regressão logística. A análise inclui pré-processamento de dados, treinamento e avaliação do modelo, e também a geração de relatórios em um documento Word.

# Descrição do Projeto
# Estrutura do Projeto

1 - Carregamento e Visualização dos Dados

Os dados são carregados a partir de um arquivo CSV (churn.csv).
São exibidas informações básicas como a estrutura do arquivo, colunas, valores ausentes e estatísticas descritivas.

2 - Análise Exploratória de Dados

Cálculo da porcentagem de clientes que permaneceram e que deixaram a empresa.
Geração de gráficos para visualizar a distribuição de churn por gênero, serviço de internet e características numéricas.

3 - Divisão dos Dados

Divisão dos dados em conjuntos de treinamento e teste.
Treinamento do Modelo

4 - Treinamento de um modelo de regressão logística.
Salvamento do modelo treinado em um arquivo (churn_model.pkl).

5 - Avaliação do Modelo
Predição nos dados de teste.
Geração de um relatório de classificação e matriz de confusão.
Salvamento dos resultados e gráficos em um documento Word (churn_analysis.docx).
Previsão em Novos Dados

6 - Criação de um arquivo de exemplo com novos clientes (new_customers.csv).
Função para carregar o modelo salvo e fazer previsões em novos dados.
Exportação das previsões para o documento Word.

# Estrutura dos Arquivos

churn.csv: Arquivo CSV contendo os dados de clientes e churn.

new_customers.csv: Arquivo CSV contendo dados de novos clientes para previsão.

churn_model.pkl: Arquivo contendo o modelo treinado.

churn_analysis.docx: Documento Word contendo os resultados da análise e previsões.

# Uso

Requisitos
Python 3.x
Bibliotecas: numpy, pandas, matplotlib, seaborn, scikit-learn, python-docx, joblib
Passos para Execução
Clone o repositório:

# Funções
predict_new_data(new_data_path)
Carrega o modelo salvo e faz previsões em novos dados.

export_predictions_to_doc(new_data, new_predictions)
Exporta as previsões para o documento Word, comparando com os resultados exploratórios.

# Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests.
