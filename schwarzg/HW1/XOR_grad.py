import numpy as np
import matplotlib.pyplot as plt

#Generate sample data
X=np.array([[0,0],[0,1],[1,0],[1,1]])
Yc=np.array([[0],[1],[1],[0]])

#setting hyper parameters
epoch_num=10000
lr=0.1
nhid=2

#Prepare functions
def sig(z):
	return 1.0/(1.0+np.exp(-z))

def dsig(z):
	sigarr=sig(z)
	return np.multiply(sigarr,(1.0-sigarr))

def ff(X,W):
	return np.dot(X,W)

def cost(Y,Yh):
	delt=-np.multiply(Yh,np.log(Y+1e-8))-np.multiply(1.-Yh,np.log(1.-Y+1e-8))
	return np.mean(delt)

def get_grad(delta,deriv,X):
	D=np.multiply(delta,deriv)
	G=np.dot(X.transpose(),D)
	return G/float(len(delta))

#Define layers = prepare weight matrices
W1=np.random.normal(size=(len(X[0])+1,nhid))
W2=np.random.normal(size=(nhid+1,len(Yc[0])))

acc=np.array([])
cst=np.array([])
for i in range(epoch_num):
	
	#feedforward
	Xin=np.concatenate((np.ones((len(X),1)),X),axis=1)
	z1=ff(Xin,W1)
	H=sig(z1)
	Hin=np.concatenate((np.ones((len(H),1)),H),axis=1)
	z2=ff(Hin,W2)
	Y=sig(z2)

	#Accuracy
	corr=np.equal(np.greater(Y,0.5),Yc).astype(int)
	acc=np.append(acc,np.mean(corr))
	cst=np.append(cst,cost(Y,Yc))

	if i%1000 is 0 : print i,cost(Y,Yc), acc[i]

	#back propagation
	err=(Y-Yc).astype(float)/((Y+1e-8)*(1.-Y+1e-8))
	gW2=get_grad(err,dsig(z2),Hin)
	
	err1=np.dot(np.multiply(err,dsig(z2)),W2[1:,:].transpose())
	gW1=get_grad(err1,dsig(z1),Xin)

	#update
	W1=W1-lr*gW1
	W2=W2-lr*gW2

#verification
Xin=np.concatenate((np.ones((len(X),1)),X),axis=1)
z1=ff(Xin,W1)
H=sig(z1)
Hin=np.concatenate((np.ones((len(H),1)),H),axis=1)
z2=ff(Hin,W2)
Y=sig(z2)

print Y
plt.plot(np.arange(epoch_num),cst)
plt.show()
	
