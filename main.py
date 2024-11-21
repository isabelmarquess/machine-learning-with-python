import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.metrics import accuracy_score, classification_report


class Modelo():
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def CarregarDataset(self, path=None):
        if path:
            names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
            self.df = pd.read_csv(path, names=names)
        else:
            iris = datasets.load_iris()
            self.df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
            self.df['Species'] = iris.target_names[iris.target]

        print(f"1. Características: {self.df.columns[:4]}")  # Mostra as 4 características
        print(f"2. Espécies: {', '.join(self.df['Species'].unique())}")  # Mostra os 3 nomes das espécies

    def TratamentoDeDados(self):
        print("Primeiras linhas do dataset:")
        print(self.df.head())

        print("\nValores ausentes em cada coluna:")
        print(self.df.isnull().sum())
        self.df.dropna(inplace=True)

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df[self.df.columns[:4]])
        plt.title("Distribuição das variáveis")
        plt.show()

        self.df['Species'] = self.df['Species'].astype('category').cat.codes

        self.df.hist(bins=10, figsize=(10, 8))
        plt.suptitle('Distribuição das variáveis numéricas')
        plt.show()

        plt.figure(figsize=(8, 6))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlação entre as variáveis')
        plt.show()

    def Treinamento(self):
        X = self.df.drop('Species', axis=1)
        y = self.df['Species']

        # Divisão em treino e teste
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Configuração dos modelos
        modelos = {
            "SVM (kernel linear)": SVC(kernel='linear'),
            "Regressão Linear": LinearRegression()
        }

        resultados = {}

        for nome, modelo in modelos.items():
            print(f"\nTreinando o modelo: {nome}")
            if nome == "Regressão Linear":
                # Para a regressão linear
                modelo.fit(self.X_train, self.y_train)
                y_pred = modelo.predict(self.X_test)
                y_pred = y_pred.round().astype(int)
                acuracia = accuracy_score(self.y_test, y_pred)
            else:
                # Para SVM
                scores = cross_val_score(modelo, self.X_train, self.y_train, cv=5)
                print(f"Acurácia média com validação cruzada ({nome}): {scores.mean():.4f}")
                modelo.fit(self.X_train, self.y_train)
                y_pred = modelo.predict(self.X_test)
                acuracia = accuracy_score(self.y_test, y_pred)
                print(f"Acurácia do modelo nos dados de teste ({nome}): {acuracia:.4f}")
                print("\nRelatório de Classificação:")
                print(classification_report(self.y_test, y_pred))
                resultados[nome] = acuracia
                
            # Comparação dos modelos
            print("\nResumo dos Modelos:")
            for nome, acuracia in resultados.items():
                print(f"{nome}: {acuracia:.4f}")
            # Escolhe o melhor modelo com base na acurácia
            melhor_modelo = max(resultados, key=resultados.get)
            print(f"\nModelo com melhor desempenho: {melhor_modelo}")
            self.model = modelos[melhor_modelo]

    def Teste(self):
        y_pred = self.model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"\nAcurácia do modelo nos dados de teste: {accuracy:.4f}")

        print("\nRelatório de Classificação:")
        print(classification_report(self.y_test, y_pred))

    def Train(self):
        self.CarregarDataset()

        self.TratamentoDeDados()

        self.Treinamento()

        self.Teste()


modelo = Modelo()
modelo.Train()
