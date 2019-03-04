import keras
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy
from keras.layers.normalization import BatchNormalization

Datas = np.array(pd.read_csv('./titanic/train.csv'))
TrainData = Datas[:801,:]
ValData = Datas[801:891,:]
TestData = np.array(pd.read_csv('./titanic/test.csv'))

Y_train = TrainData[:,1]
Y_val = ValData[:,1]

X_train = np.zeros((801,6),dtype='float32');
X_val = np.zeros((90,6),dtype='float32');
X_test = np.zeros((418,6),dtype='float32');

X_train[:,0] = TrainData[:,2]
X_train[:,1] = np.float32(TrainData[:,4]=='male')
X_train[:,2] = np.array(map(lambda x : 0. if(np.isnan(x)) else x, list(TrainData[:,5])))
X_train[:,3:5] = TrainData[:,6:8]
X_train[:,5] = TrainData[:,9]

X_val[:,0] = ValData[:,2]
X_val[:,1] = np.float32(ValData[:,4]=='male')
X_val[:,2] = np.array(map(lambda x : 0. if(np.isnan(x)) else x, list(ValData[:,5])))
X_val[:,3:5] = ValData[:,6:8]
X_val[:,5] = ValData[:,9]

X_test[:,0] = TestData[:,1]
X_test[:,1] = np.float32(TestData[:,3]=='male')
X_test[:,2] = np.array(map(lambda x : 0. if(np.isnan(x)) else x, list(TestData[:,4])))
X_test[:,3:5] = TestData[:,5:7]
X_test[:,5] = np.array(map(lambda x : 0. if(np.isnan(x)) else x, list(TestData[:,8])))

norm = np.array([3.,1.,80.,8.,6.,512.])
X_train = X_train/norm
X_val = X_val/norm
X_test = X_test/norm

n_hidden = [50,25,10]

model = Sequential()
for i in range(3):
    model.add(Dense(n_hidden[i],kernel_initializer='he_normal',kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss=binary_crossentropy,optimizer=Adam(lr=0.0001,beta_1=0.9,beta_2=0.999),metrics=['accuracy'])

history = model.fit(X_train,Y_train,epochs=500,batch_size=100,verbose=1)
'''
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
'''
loss_and_metrics = model.evaluate(X_val,Y_val)
print(loss_and_metrics)

pred = model.predict_classes(X_test,batch_size=1)
f = open('prediction.csv','w')
wr = csv.writer(f)
wr.writerow(['PassengerId','Survived'])
for i in range(418):
    wr.writerow([TestData[i,0],pred[i,0]])





