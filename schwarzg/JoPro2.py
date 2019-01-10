from math import exp,sqrt,log
import numpy as np
from mnist import MNIST
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

mndata=MNIST('./')

images,labels=mndata.load_training()

images=np.array(images)

maxval=float(np.amax(images))

images=images/maxval

test=images[0].reshape(28,28)
test=np.concatenate((test,images[1].reshape(28,28)),axis=1)

plt.imsave('sampleinput.png',test*maxval,cmap="gray")


lr=0.001	#learningrate

#foward function 
def foward(V,W,B):
	dE=-np.dot(V,W)-B
	P=np.random.uniform(size=(len(V),len(W[0])))
	H=np.zeros(np.shape(P))
	for i in range(len(P)):
		for j in range(len(P[0])):
			H[i][j]=1.0 if 1./(1.+exp(dE[i][j]))>P[i][j] else 0.0
	return H

def backward(H,W,A):
	dE=-np.dot(H,W.transpose())-A
	P=np.random.uniform(size=(len(H),len(W)))
	V=np.zeros(np.shape(P))
	for i in range(len(P)):
		for j in range(len(P[0])):
			V[i][j]=1.0 if 1./(1.+exp(dE[i][j]))>P[i][j] else 0.0
	return V

def Free_energy(V,H,W,A,B):
	E=-np.sum(np.multiply(np.dot(V,W),H),axis=1)-np.sum(np.multiply(V,A),axis=1)-np.sum(np.multiply(H,B),axis=1)
	Z=np.sum(np.exp(-E))
	return -log(Z)	


nh=50
W=0.5*np.random.normal(size=(784,nh))
A=np.random.normal(size=(1,784))
B=np.random.normal(size=(1,nh))

epoch=3
batch_size=1
batch_list=np.arange(len(images))

for i in range(epoch):

	np.random.shuffle(batch_list)
	Vt=images[0:100,:]
	for j in range(len(batch_list)/batch_size):

		V=images[batch_list[j*batch_size:(j+1)*batch_size],:]
		H=foward(V,W,B)
		V1=backward(H,W,A)
		H1=foward(V1,W,B)

		gW=np.dot(V.transpose(),H)-np.dot(V1.transpose(),H1)
		gA=np.sum(V-V1,axis=0)
		gB=np.sum(H-H1,axis=0)
	
		W=W+lr/sqrt(batch_size)*gW
		A=A+lr/sqrt(batch_size)*gA
		B=B+lr/sqrt(batch_size)*gB
		if j%100==99:
			print 'epoch :', i, ', batch :', j, ', F =',Free_energy(Vt,foward(Vt,W,B),W,A,B)
	
Vout=[]
for j in range(10):
	#Gibbs Sampling
	H=np.random.uniform(0,1,size=(1,nh))
	for i in range(100):
		V=backward(H,W,A)
		H=foward(V,W,B)
	if len(Vout)==0 : Vout=V.reshape(28,28)
	else : Vout=np.concatenate((Vout,V.reshape(28,28)),axis=1)
	
plt.imsave('sampleoutput6.png',Vout*maxval,cmap="gray")
	
