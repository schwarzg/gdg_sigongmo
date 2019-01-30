import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0],784)
X_test = X_test.reshape(X_test.shape[0],784)
n_hidden = 1000
n_out = 10

Y_train = keras.utils.to_categorical(Y_train,n_out)
Y_test = keras.utils.to_categorical(Y_test,n_out)

model = Sequential()
model.add(Dense(n_hidden,input_dim=784,activation='sigmoid'))
model.add(Dense(n_out,activation='softmax'))

model.compile(loss=categorical_crossentropy,optimizer=SGD(lr=0.01),metrics=['accuracy'])

epochs = 50
batch_size = 100

model.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size)

loss_and_metrics = model.evaluate(X_test,Y_test)
print(loss_and_metrics)

