import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from tensorflow import keras

# tensorflow tester
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

f = lambda x: 10 + 20 * x^2 + 30 * x^3

def make_model(input_shape):
    ret = Sequential()
    ret.add(Dense(10, input_shape=input_shape, activation='sigmoid'))
    # ret.add(Dense(3, input_shape=input_shape, activation='sigmoid'))
    # ret.add(Dense(3, activation='relu'))
    ret.add(Dense(1, activation='linear'))
    return ret

X = np.arange(0, 300)
np.random.shuffle(X)
Y = f(X)

X_scaled = preprocessing.scale(X)
# scaler = preprocessing.StandardScaler().fit(X)
# X = scaler.transform(X)

input_shape = X.shape[1:]

model = make_model([1])
model.compile(loss='mse', optimizer='RMSprop', metrics=['mean_squared_error'])
model.fit(X, Y, epochs=100, batch_size=10, verbose=1, validation_split=0.2)
