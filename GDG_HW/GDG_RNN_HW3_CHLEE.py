'''
###########################################################
GDG RNN STUDY HW#3
Name : Chang Hyun Lee
Date : 2019.1.31.
###########################################################
Content : MNIST Classification using ANN - tensorflow
###########################################################
'''

'''
###########################################################
항상 이 순서를 기억하기
1. 데이터 셋 불러오고 정규화하기
2. 모델 정의하기
3. 손실함수와 Optimizer 정의하기
4. 모델 학습
5. 모델 검증 및 평가
###########################################################
'''


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import numpy as np

###########################################################
#1. 데이터 셋 불러오고 정규화하기
###########################################################
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0 # (60000,784)
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0 # (10000, 784)

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

X_valid, X_train = X_train[:5000], X_train[5000:] #인덱싱해서 validation set 만들기
y_valid, y_train = y_train[:5000], y_train[5000:] # 처음 5000개와 5000개 이후부터 나머지까지 인덱싱



###########################################################
#2. 모델 정의하기
###########################################################
n_in = 784
n_hidden = 200
n_out = 10

model = Sequential()
model.add(Dense(n_hidden, input_dim=n_in))
model.add(Activation('sigmoid'))

model.add(Dense(n_out))
model.add(Activation('softmax'))



###########################################################
#3. 손실함수와 Optimizer 정의하기
###########################################################
model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])



###########################################################
#4. 모델 학습하기
###########################################################
epochs = 1000
batch_size = 100

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)



###########################################################
#5. 모델 검증 및 평가하기
###########################################################
loss_and_metrics = model.evaluate(X_test,y_test)


