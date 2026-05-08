from sklearn.naive_bayes import BernoulliNB # o algoritmo de aprendizado
from sklearn.feature_extraction.text import CountVectorizer # quantificar o texto pra números
from sklearn.model_selection import train_test_split # divisão de dados pra teste e treino
from sklearn.metrics import classification_report # fala os parâmetros de acurácia do modelo
from sklearn.model_selection import cross_val_score
import glob
import re

# Caminho da pasta onde está o csv
path = "csvfolder"
files = glob.glob(path + "/*.csv")

# Define sets de data e labels (X e y) pro algoritmo treinar
data = [] # guarda o texto
labels = [] # guarda os resultados (0 ou 1)
for file in files: # vai fazer a operação pra todos os .csv da pasta
    with open(file, 'r', encoding='utf-8', errors='ignore') as f: # abre o arquivo e depois o fecha guardando a informação dele como o objeto "f"
        for line in f:
            line = line.strip() # limpa coisas de espaço e pular linha e tal pra n dar problema

            if not line:
                continue

            # só mantém as linhas que são separadas de forma correta pelo "tab", dados válidos
            if '\t' in line:
                text, label = line.rsplit('\t', 1) # separa o texto e as labels de uma forma que o programa compreenda

                if label in ['0', '1']: # só valores válidos, como usamos binário nesse modelo é 0 e 1
                    data.append(text)
                    labels.append(int(label))
# pré-processamento dos dados para padronizar eles antes de vetorizar
data_limpa = []

for texto in data:
    texto = texto.lower() # letra minuscula
    texto = re.sub('[^a-zA-Z]', ' ', texto) # remover pontuação
    data_limpa.append(texto)

# essa parte basicamente pega as palavras e "aprende" e transforma em números no caso do X
vectorizer = CountVectorizer(
    binary=True, # considera apenas números binários que é o que usamos no modelo
    stop_words='english', # ignora palavras comuns de conexão em inglês
    ngram_range=(1, 2), # pega palavras únicas e pares de palaras
    min_df=2, # ignora palavras raras
    max_features=2000 # número máximo das palavras mais usadas, tipo um top 2000
)
X = vectorizer.fit_transform(data)
y = labels

# divide os dados entre quantidade pra treinar e quantidade pra testar, tamanho definido pelo argumento test_size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# cria o modelo e o "fit" basicamente pega o que é treinado e aprende o que são palavras boas e palavras ruins
bnb = BernoulliNB(alpha=0.5, class_prior=[0.5, 0.5])
bnb.fit(X_train, y_train)
# prever a label de dados novos basicamente
y_pred = bnb.predict(X_test)
# roda o progrmaa 5 vezes e tira a média dos resultados, dando uma visão mais precisa da performance de acurácia do modelo
scores = cross_val_score(bnb, X_train, y_train, cv=5)
print(scores)
print(scores.mean())
# fala o resultado da performance do modelo
print(classification_report(y_test, y_pred))