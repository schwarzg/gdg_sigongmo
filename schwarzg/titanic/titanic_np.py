import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import J_ml.activation as act

#Process input data and answer
df_tr=pd.read_csv("train.csv")
df_te=pd.read_csv("test.csv")

df_tr=df_tr[["Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
df_tr=df_tr.dropna(how="any")
df_tr=df_tr.replace({"male":0,"female":1,"Q":0,"S":1,"C":2})
df_tr=df_tr.as_matrix()
X_t=df_tr[:,1:]
Y_t=df_tr[:,0]
ninp=len(X_t[0])
ndat=len(Y_t)
Y_t=Y_t.reshape((ndat,1))

df_te=df_te[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
df_te=df_te.dropna(how="any")
df_te=df_te.replace({"male":0,"female":1,"Q":0,"S":1,"C":2})
X_v=df_te.as_matrix()


#hyper parameters
lr=0.01	#learningrate
epoch_num=1000000

#Weight matrix
W1=np.random.normal(size=(ninp+1,ninp/2)) #W1[0] is for bias
W2=np.random.normal(size=(ninp/2+1,1)) #W2[0] is for bias

#feedfoward function 
def feedfoward(X,W,B):
	return np.dot(X,W)+B

def cost(Y,Yh):
	delt=-np.multiply(Yh,np.log(Y+1e-8))-np.multiply(1.0-Yh,np.log(1.0-Y+1e-8))
	return np.mean(delt)

def get_gradient(delta,deriv,X):
	D=np.multiply(delta,deriv)
	G=np.dot(X.transpose(),D)
	return G/float(len(delta))

acc=np.array([])
cst=np.array([])
for i in range(epoch_num):

	#feedfoward
	Xin=np.concatenate((np.ones((ndat,1)),X_t),axis=1)
	z1=feedfoward(Xin,W1,0)
	H=act.sig(z1)
	Hin=np.concatenate((np.ones((ndat,1)),H),axis=1)
	z2=feedfoward(Hin,W2,0)
	Y=act.sig(z2)
	
	corr=np.equal(np.greater(Y,0.5),Y_t).astype(int)
	acc=np.append(acc,np.mean(corr))
	cst=np.append(cst,cost(Y,Y_t))


	if i%10000 is 0 :
		print i, cst[i], acc[i]	
	
	#back propagation, gradient calculation
	#delta=Y-Yh
	delta=(Y-Y_t).astype(float)/((Y+1e-8)*(1.0-Y+1e-8))
	gW2=get_gradient(delta,act.sigd(z2),Hin)

	delta2=np.dot(np.multiply(delta,act.sigd(z2)),W2[1:,:].transpose())
	
	gW1=get_gradient(delta2,act.sigd(z1),Xin)
	
	#update
	W1=W1-lr*gW1
	W2=W2-lr*gW2

print "does it run?"

#plt.plot(np.arange(epoch_num),cst)
plt.plot(np.arange(epoch_num),acc)
plt.show()
