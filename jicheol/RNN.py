import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
from keras.layers.recurrent import SimpleRNN,LSTM,GRU

def sin(x,T=100):
	return np.sin(2.0*np.pi*x/T)

def toy_problem(T=100,ampl=0.05):
	x=np.arange(0,2*T+1)
	noise=ampl*np.random.uniform(low=-1.0,high=1.0,size=len(x))
	return sin(x)+noise

T=100
f=toy_problem(T)
length_of_sequences=2*T
maxlen=25

data=[]
target=[]

for i in range(0,length_of_sequences-maxlen+1):
	data.append(f[i:i+maxlen])
	target.append(f[i+maxlen])

X=np.array(data).reshape(len(data),maxlen,1)
Y=np.array(target).reshape(len(data),1)

X=np.zeros((len(data),maxlen,1),dtype=float)
Y=np.zeros((len(data),1),dtype=float)

for i,seq in enumerate(data):
	for t,value in enumerate(seq):
		X[i,t,0]=value
	Y[i,0]=target[i]

N_train=int(len(data)*0.9)
N_validation=len(data)-N_train

X_train = X[0:N_train,:]
X_validation = X[N_train:-1,:]
Y_train = Y[0:N_train,:]
Y_validation = Y[N_train:-1,:]

n_in = len(X[0][0])
n_hidden = 20
n_out = len(Y[0])

model=Sequential()
model.add(LSTM(n_hidden,kernel_initializer='lecun_uniform',input_shape=(maxlen,n_in)))
model.add(Dense(n_out,kernel_initializer='lecun_uniform'))
model.add(Activation('linear'))

model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.001,beta_1=0.9,beta_2=0.999))

early_stopping=EarlyStopping(monitor='val_loss',patience=10,verbose=1)
model.fit(X_train,Y_train,epochs=500,batch_size=15,\
	validation_data=(X_validation,Y_validation),verbose=2,callbacks=[early_stopping])

truncate=maxlen
Z=X[:1]

original=[f[i] for i in range(maxlen)]
predicted=[None for i in range(maxlen)]

for i in range(length_of_sequences-maxlen+1):
	z_=Z[-1:]
	y_=model.predict(z_)
	sequence_=np.concatenate((z_.reshape(maxlen,n_in)[1:],y_),axis=0).reshape(1,maxlen,n_in)
	Z=np.append(Z,sequence_,axis=0)
	predicted.append(y_.reshape(-1))

plt.plot(toy_problem(T,ampl=0),linestyle='dotted')
plt.plot(original,label='original')
plt.plot(predicted,label='predicted')
plt.legend(loc='best')
plt.xlabel('time')
plt.show()








