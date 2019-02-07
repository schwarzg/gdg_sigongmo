# -*- coding: utf-8 -*-
#Prepare package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import J_ml.activation as act


#Process input data and answer
df_tr=pd.read_csv("train.csv")


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


df_tr['Initial']= df_tr.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations

df_tr['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
												['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

df_tr.loc[(df_tr.Age.isnull())&(df_tr.Initial=='Mr'),'Age'] = 33
df_tr.loc[(df_tr.Age.isnull())&(df_tr.Initial=='Mrs'),'Age'] = 36
df_tr.loc[(df_tr.Age.isnull())&(df_tr.Initial=='Master'),'Age'] = 5
df_tr.loc[(df_tr.Age.isnull())&(df_tr.Initial=='Miss'),'Age'] = 22
df_tr.loc[(df_tr.Age.isnull())&(df_tr.Initial=='Other'),'Age'] = 46

df_tr['Embarked'].fillna('S', inplace=True)

df_trs=df_tr[["Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
df_trs=df_trs.replace({"male":0,"female":1,"Q":0,"S":1,"C":2})

X_t=df_trs[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]].as_matrix()
Y_t=df_trs[["Survived"]].as_matrix()
ninp=len(X_t[0])
ndat=len(Y_t)

#Weight matrix
W1=np.sqrt(2.0/float(ninp))*np.random.normal(size=(ninp+1,100)) #W1[0] is for bias
W2=np.sqrt(2.0/float(100.0))*np.random.normal(size=(101,1)) #W2[0] is for bias

#hyper parameters
lr=0.001

epoch_num=100000
perc=epoch_num/100

acc=np.array([])
cst=np.array([])
m1=np.zeros(W1.shape)
v1=np.zeros(W1.shape)
m2=np.zeros(W2.shape)
v2=np.zeros(W2.shape)
b1=0.9
b2=0.999

print "Epoch, Cost, Acc"
for i in range(epoch_num+1):

	#feedfoward
	Xin=np.concatenate((np.ones((ndat,1)),X_t),axis=1)
	z1=feedfoward(Xin,W1,0)
	H=act.relu(z1)
	Hin=np.concatenate((np.ones((ndat,1)),H),axis=1)
	z2=feedfoward(Hin,W2,0)
	Y=act.sig(z2)
	
	corr=np.equal(np.greater(Y,0.5),Y_t).astype(int)
	acc=np.append(acc,np.mean(corr))
	cst=np.append(cst,cost(Y,Y_t))

	if i%perc is 0 :
		print i, cst[i], acc[i]
	if i is epoch_num :
		break
	
	#back propagation, gradient calculation
	deltao=(Y-Y_t).astype(float)/((Y+1e-8)*(1.0-Y+1e-8))
	gW2=get_gradient(deltao,act.sigd(z2),Hin)

	deltah=np.dot(np.multiply(deltao,act.sigd(z2)),W2[1:,:].transpose())	
	gW1=get_gradient(deltah,act.relud(z1),Xin)
	
	#Adam
	m1=b1*m1+(1-b1)*gW1
	v1=b2*v1+(1-b2)*np.square(gW1)	
	m1_h=m1/(1-b1**(i+1))
	v1_h=v1/(1-b2**(i+1))

	m2=b2*m2+(1-b1)*gW2
	v2=b2*v2+(1-b2)*np.square(gW2)	
	m2_h=m2/(1-b1**(i+1))
	v2_h=v2/(1-b2**(i+1))
	
	#update
	W1=W1-lr*np.divide(m1,np.sqrt(v1)+1e-8)
	W2=W2-lr*np.divide(m2,np.sqrt(v2)+1e-8)

'''
plt.xlabel("epoch") 
plt.ylabel("Training accuracy")
plt.plot(np.arange(epoch_num+1),acc)
plt.show()
'''

df_te=pd.read_csv("test.csv")


df_te['Initial']= df_te.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations

df_te['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
												['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

df_te.loc[(df_te.Age.isnull())&(df_te.Initial=='Mr'),'Age'] = 33
df_te.loc[(df_te.Age.isnull())&(df_te.Initial=='Mrs'),'Age'] = 36
df_te.loc[(df_te.Age.isnull())&(df_te.Initial=='Master'),'Age'] = 5
df_te.loc[(df_te.Age.isnull())&(df_te.Initial=='Miss'),'Age'] = 22
df_te.loc[(df_te.Age.isnull())&(df_te.Initial=='Other'),'Age'] = 46

avF=df_te['Fare'].mean()
df_te['Fare'].fillna(np.float32(avF), inplace=True)

df_te=df_te[["PassengerId","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
df_te=df_te.replace({"male":0,"female":1,"Q":0,"S":1,"C":2})
id=df_te[["PassengerId"]].as_matrix()
X_v=df_te[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]].as_matrix()

#feedfoward
ndat=len(X_v)
Xin=np.concatenate((np.ones((ndat,1)),X_v),axis=1)
z1=feedfoward(Xin,W1,0)
H=act.relu(z1)
Hin=np.concatenate((np.ones((ndat,1)),H),axis=1)
z2=feedfoward(Hin,W2,0)
Y_v=act.sig(z2)

A=np.greater(Y_v,0.5).astype(int)
sub=np.concatenate((id,A),axis=1)
sub=pd.DataFrame(sub,columns=['PassengerId','Survived'])

sub.to_csv('submission.csv',index=False)

