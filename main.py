# Importando bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# Carregando os dados
data = pd.read_csv('Dados_RH_Turnover.csv', sep=';')

# Visualizando as primeiras linhas do dataset
print(data.head())

# Definindo variáveis preditoras e variável alvo
X = data.drop('<Nome_da_Coluna_Alvo>', axis=1)  # Substitua '<Nome_da_Coluna_Alvo>' pelo nome correto da coluna alvo
y = data['<Nome_da_Coluna_Alvo>']                # Substitua '<Nome_da_Coluna_Alvo>' pelo nome correto da coluna alvo

#Colunas:NivelSatisfacao ; UltimaAvaliacao ; NumeroProjetos ; MediaHorasMensais ; AnosTrabalhoEmpresa ; ProblemaColega ; SaiuDaEmpresa ; PromocaoCargo_UltimoAno ; DeptoAtuacao ; Salario

# Verificando os valores únicos na variável alvo
print("Valores únicos na variável alvo antes da conversão:", y.unique())

# Convertendo valores contínuos em uma classe binária, se necessário
# Aqui, assumimos que valores maiores que 0.5 representam "turnover" (1) e valores de 0.5 ou menos representam "não turnover" (0).
y = y.apply(lambda x: 1 if x > 0.5 else 0)

# Confirmando a conversão
print("Valores únicos na variável alvo após a conversão:", y.unique())

# Convertendo variáveis categóricas em variáveis dummy
X = pd.get_dummies(X, drop_first=True)

# Dividindo o dataset em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Padronizando os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dicionário para armazenar os modelos e suas métricas
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Neural Network": MLPClassifier(max_iter=500)
}

# Treinando os modelos e avaliando suas acurácias
accuracies = {}
conf_matrices = {}

for model_name, model in models.items():
    # Treinando o modelo
    model.fit(X_train, y_train)
    
    # Realizando previsões
    y_pred = model.predict(X_test)
    
    # Calculando a acurácia e a matriz de confusão
    accuracies[model_name] = accuracy_score(y_test, y_pred)
    conf_matrices[model_name] = confusion_matrix(y_test, y_pred)

# Exibindo a acurácia de cada modelo
print("Acurácia dos modelos:")
for model_name, accuracy in accuracies.items():
    print(f"{model_name}: {accuracy:.2f}")

# Visualizando as matrizes de confusão
for model_name, conf_matrix in conf_matrices.items():
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.show()
