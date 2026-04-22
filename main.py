from sklearn.naive_bayes import BernoulliNB # o algoritmo de aprendizado
from sklearn.feature_extraction.text import CountVectorizer # quantificar o texto pra números
from sklearn.model_selection import train_test_split # divisão de dados pra teste e treino
from sklearn.metrics import classification_report # fala os parâmetros de acurácia do modelo
import glob

# Caminho da pasta onde está o csv
path = "csvfolder"
files = glob.glob(path + "/*.csv")

# Define sets de data e labels (X e y) pro algoritmo treinar
data = [] # guarda o texto
labels = [] # guarda os resultados (0 ou 1)

with open('csvfolder/imdb_labelled.csv', 'r', encoding='utf-8', errors='ignore') as f: # abre o arquivo e depois o fecha guardando a informação dele como o objeto "f"
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

# essa parte basicamente pega as palavras e "aprende" e transforma em números no caso do X
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(data)
y = labels

# divide os dados entre quantidade pra treinar e quantidade pra testar, tamanho definido pelo argumento test_size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# cria o modelo e o "fit" basicamente pega o que é treinado e aprende o que são palavras boas e palavras ruins
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
# prever a label de dados novos basicamente
y_pred = bnb.predict(X_test)
# fala o resultado da performance do modelo
print(classification_report(y_test, y_pred))