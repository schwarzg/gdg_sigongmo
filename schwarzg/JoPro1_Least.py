import numpy as np
from mnist import MNIST

mndata=MNIST('./')

images,labels=mndata.load_training()
timages,tlabels=mndata.load_testing()
images=np.array(images)
timages=np.array(timages)

#one-hot-vectorize
tmp=np.zeros((len(labels),10))
for i in range(len(labels)):
	tmp[i][labels[i]]=1

labels=tmp

tmp=np.zeros((len(tlabels),10))
for i in range(len(tlabels)):
	tmp[i][tlabels[i]]=1

tlabels=tmp
del tmp

lr=0.01	#learningrate

W1=np.random.normal(size=(785,500)) #W1[0] is for bias
W2=np.random.normal(size=(501,10)) #W2[0] is for bias

'''
W1=np.random.normal(size=(784,500))
b1=np.random.normal(size=(1,500))

W2=np.random.normal(size=(500,10))
b2=np.random.normal(size=(1,10))

W1=np.zeros((784,500))
b1=np.zeros((1,500))

W2=np.zeros((500,10))
b2=np.zeros((1,10))
'''
def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def dsigmoid(z):
	sigarr=sigmoid(z)
	return np.multiply(sigarr,(1.0-sigarr))

#feedfoward function 
def feedfoward(X,W,B):
	return np.dot(X,W)+B

def cost(Y,Yh):
	delt=0.5*np.square(Y-Yh)
	return np.mean(delt)

def get_gradient(delta,deriv,X):
	D=np.multiply(delta,deriv)
	G=np.dot(X.transpose(),D)
	return G/float(len(delta))

batch_num=1000
for i in range(10000):

	batch_list=np.random.randint(60000,size=batch_num)
	X=images[batch_list,:]
	Yh=labels[batch_list,:]

	#feedfoward
	Xin=np.concatenate((np.ones((batch_num,1)),X),axis=1)
	z1=feedfoward(Xin,W1,0)
	H=sigmoid(z1)
	Hin=np.concatenate((np.ones((batch_num,1)),H),axis=1)
	z2=feedfoward(Hin,W2,0)
	Y=sigmoid(z2)
	
	corr=np.equal(np.argmax(Y,axis=1),np.argmax(Yh,axis=1)).astype(int)
	accuracy=np.mean(corr)

	print i, cost(Y,Yh), accuracy	
	
	#back propagation, gradient calculation
	delta=Y-Yh

	gW2=get_gradient(delta,dsigmoid(z2),Hin)

	delta2=np.dot(np.multiply(delta,dsigmoid(z2)),W2[1:,:].transpose())
	
	gW1=get_gradient(delta2,dsigmoid(z1),Xin)
	
	#update
	W1=W1-lr*gW1
	W2=W2-lr*gW2
	
print "does it run?"

