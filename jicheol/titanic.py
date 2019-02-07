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

TrainData = np.array(pd.read_csv('./titanic/train.csv'))
ValData = np.array(pd.read_csv('./titanic/test.csv'))
TestData = np.array(pd.read_csv('./titanic/gender_submission.csv'))

Y_train = TrainData[:,1]
Y_test = TestData[:,1]

X_train = np.zeros((891,6),dtype='float32');
X_val = np.zeros((418,6),dtype='float32');

X_train[:,0] = TrainData[:,2]
X_train[:,1] = np.float32(TrainData[:,4]=='male')
X_train[:,2] = np.array(map(lambda x : 0. if(np.isnan(x)) else x, list(TrainData[:,5])))
X_train[:,3:5] = TrainData[:,6:8]
X_train[:,5] = TrainData[:,9]

X_val[:,0] = ValData[:,1]
X_val[:,1] = np.float32(ValData[:,3]=='male')
X_val[:,2] = np.array(map(lambda x : 0. if(np.isnan(x)) else x, list(ValData[:,4])))
X_val[:,3:5] = ValData[:,5:7]
X_val[:,5] = np.array(map(lambda x : 0. if(np.isnan(x)) else x, list(ValData[:,8])))

norm = np.array([3.,1.,80.,8.,6.,512.])
X_train = X_train/norm
X_val = X_val/norm

n_hidden = 500

model = Sequential()
model.add(Dense(n_hidden,kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss=binary_crossentropy,optimizer=Adam(lr=0.001,beta_1=0.9,beta_2=0.999),metrics=['accuracy'])

history = model.fit(X_train,Y_train,epochs=100,batch_size=25,verbose=0)

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

loss_and_metrics = model.evaluate(X_val,Y_test)
print(loss_and_metrics)

pred = model.predict_classes(X_val,batch_size=1)
f = open('prediction.csv','w')
wr = csv.writer(f)
wr.writerow(['PassengerId','Survived'])
for i in range(418):
    wr.writerow([ValData[i,0],pred[i,0]])





