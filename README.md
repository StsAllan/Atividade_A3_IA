Descrição do Projeto: Análise e Modelagem Preditiva de Turnover de Funcionários

Neste projeto, o objetivo é desenvolver e comparar modelos de classificação para prever o turnover de funcionários com base em um conjunto de dados de Recursos Humanos. Essa análise tem como finalidade identificar os fatores que mais contribuem para o desligamento de colaboradores, fornecendo insights que possam ajudar a empresa a reter talentos e reduzir custos associados ao turnover.

Passos Realizados:
Exploração dos Dados: Leitura e visualização inicial do dataset “Dados_RH_Turnover.csv” para entender a estrutura dos dados e as variáveis disponíveis.
Pré-processamento dos Dados: Transformação das variáveis categóricas em variáveis numéricas por meio de "one-hot encoding" e padronização dos dados para melhorar a performance dos algoritmos.
Modelagem: Foram aplicados diversos algoritmos de classificação:
Árvore de Decisão: Simples e fácil de interpretar, cria divisões baseadas em variáveis decisivas para o turnover.
K-Nearest Neighbors (K-NN): Classifica com base nos funcionários mais próximos, considerando características similares.
Naive Bayes: Modelo probabilístico, útil para dados com variáveis independentes.
Regressão Logística: Classificador baseado em probabilidade, ideal para variáveis binárias.
Rede Neural: Modelo robusto para aprendizado complexo, capturando padrões não-lineares dos dados.
Avaliação dos Modelos: Cada modelo foi avaliado em termos de acurácia e matriz de confusão para determinar a taxa de acertos e os erros de classificação.
Resultados:
O modelo de Rede Neural apresentou a maior acurácia entre os algoritmos testados, indicando maior capacidade de generalização. As matrizes de confusão também foram analisadas para verificar o desempenho na classificação correta de turnover (positivos) e retenção (negativos).

