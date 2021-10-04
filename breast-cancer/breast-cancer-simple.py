import pandas as pd

entradas = pd.read_csv("entradas_breast.csv")
saidas = pd.read_csv("saidas_breast.csv")

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(entradas, saidas, test_size = 0.25)

import keras
from keras.models import Sequential
from keras.layers import Dense
classificador = Sequential()
# number of units = (number of inputs(30) + number of outputs(1))/2
classificador.add(Dense(units = 16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
classificador.add(Dense(units = 1, activation='sigmoid'))

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10,
                  epochs = 100)