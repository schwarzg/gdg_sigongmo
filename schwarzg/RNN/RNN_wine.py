# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import csv
import itertools
import nltk


#initial setting : prepare data
vocabulary_size=1000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

df = pd.read_csv("winemag-data-130k-v2.csv")
df = df["description"]

train_string=df.str.cat()

train_string=train_string.decode('utf-8').lower()

# Split full comment into sentences
sentences=nltk.sent_tokenize(train_string)[:10]

# Append SENTENCE_START and SENTENCE_END
sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]

print "Parsed %d sentences." % (len(sentences))	

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

print "\nExample sentence: '%s'" % sentences[0]
print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]

X_tr=[]
for sent in tokenized_sentences:
	X_tr=X_tr+[word_to_index[w] for w in sent] 
print X_tr[:1000]
tlen=len(X_tr)



#Preprocess data
itau=5
otau=1
idim=len(word_to_index)
print idim
x_tr=[]
y_tr=[]
for i in range(0,tlen-tau):
	x_tr.append(X_tr[i:i+itau])
	y_tr.append(X_tr[i+itau])
X_tr=np.array(x_tr).reshape(len(x_tr),itau)
Y_tr=np.array(y_tr).reshape(len(y_tr),otau)
Ndat=len(x_tr)

print X_tr.shape
print Y_tr.shape

def ohe(x,vsize):
	ohev=np.zeros((len(x),len(x[0]),vsize))
	for i in range(len(x)):
		for j in range(len(x[0])):
			ohev[i,j,x[i,j]]=1
	return ohev

xo=ohe(X_tr[:2,:],idim)
yo=ohe(Y_tr[:2,:],idim)
print xo.shape,yo.shape

#Construct Network
#define activation functions as hypertanget function
def act(x):
	return np.tanh(x)
def dact(x):
	return 1.0-np.square(np.tanh(x))

def softmax(x):
	shiftx = x - np.max(x)
	exps = np.exp(shiftx)
	return exps / np.sum(exps)

	

#Feedfoward though time. the dimension of h should satisfy (tlen+1,hidden dim)
def feedfoward(x,U,W,V,b,c):
	Ndat=len(x)
	h=np.zeros((Ndat,tau+1,hdim)) #h[-1]=h[tau] means hidden in t=-1, set to zero
	z=np.zeros(h.shape)
	y=np.zeros((Ndat,1,idim))
	for i in range(0,tau):
		z[:,i,:]=np.dot(x[:,i,:],U)+np.dot(h[:,i-1,:],W)+b
		h[:,i,:]=act(z[:,i,:])
	#print i,"z\n",z
	#print i,"h\n",h
	y[:,0,:]=np.dot(h[:,tau-1,:],V)+c
	#print "y\n",y
	return [y,z,h]


#cost function : mean square error
def cost_ms(y,yt):
	return 0.5*np.mean(np.square(y-yt))


#Backpropagation through time
def BPTT(x,yt,U,W,V,b,c):
	Ndat=len(x)
	y,z,h=feedfoward(x,U,W,V,b,c)
	dEdU=np.zeros(U.shape)
	dEdW=np.zeros(W.shape)
	dEdV=np.zeros(V.shape)
	dEdb=np.zeros(b.shape)
	dEdc=np.zeros(c.shape)
	del_o=(y-yt)[:,0,:]
	print del_o.shape
	t=tau
	dEdV+=np.dot(h[:,t-1,:].transpose(),del_o)/Ndat
	dEdc+=np.mean(del_o,axis=0)
	del_h=np.multiply(np.dot(del_o,V.transpose()),1-h[:,t-1,:]**2)
	for i in range(t)[::-1]:
		dEdW+=np.dot(h[:,i-1,:].transpose(),del_h)/Ndat
		dEdU+=np.dot(x[:,i,:].transpose(),del_h)/Ndat
		dEdb+=np.mean(del_h,axis=0)
		del_h=np.multiply(np.dot(del_h,W.transpose()),1-h[:,i-1,:]**2)
	return [dEdU,dEdW,dEdV,dEdb,dEdc]

#Check gradient
def num_grad(x,y_t,U,W,V,b,c):
	Ndat=len(x)
	y,_,_=feedfoward(x,U,W,V,b,c)
	cost=cost_ms(y,y_t)
	ngU=np.zeros_like(U)	
	ngW=np.zeros_like(W)	
	ngV=np.zeros_like(V)	
	ngb=np.zeros_like(b)	
	ngc=np.zeros_like(c)

	d=1e-5
	for (i,j), v in np.ndenumerate(U):
		dU=np.array(U)
		dU[i,j]=v+d
		yd,_,_=feedfoward(x,dU,W,V,b,c)
		dcost=cost_ms(yd,y_t)
		ngU[i,j]=(dcost-cost)/d
	for (i,j), v in np.ndenumerate(W):
		dW=np.array(W)
		dW[i,j]=v+d
		yd,_,_=feedfoward(x,U,dW,V,b,c)
		dcost=cost_ms(yd,y_t)
		ngW[i,j]=(dcost-cost)/d
	for (i,j), v in np.ndenumerate(V):
		dV=np.array(V)
		dV[i,j]=v+d
		yd,_,_=feedfoward(x,U,W,dV,b,c)
		dcost=cost_ms(yd,y_t)
		ngV[i,j]=(dcost-cost)/d
	for (i,j), v in np.ndenumerate(b):
		db=np.array(b)
		db[i,j]=v+d
		yd,_,_=feedfoward(x,U,W,V,db,c)
		dcost=cost_ms(yd,y_t)
		ngb[i,j]=(dcost-cost)/d
	for (i,j), v in np.ndenumerate(c):
		dc=np.array(c)
		dc[i,j]=v+d
		yd,_,_=feedfoward(x,U,W,V,b,dc)
		dcost=cost_ms(yd,y_t)
		ngc[i,j]=(dcost-cost)/d
	
	return [ngU,ngW,ngV,ngb,ngc]


	
#define weight matrices, hidden nodes
hdim=idim*2
U=0.01*np.random.normal(size=(idim,hdim))
W=0.01*np.random.normal(size=(hdim,hdim))
V=0.01*np.random.normal(size=(hdim,idim))
b=0.01*np.random.normal(size=(1,hdim))
c=0.01*np.random.normal(size=(1,idim))

y,z,h=feedfoward(xo,U,W,V,b,c)
print "AfterFeedFoward", y.shape,z.shape,h.shape

#gradiend check
#y,_,_=feedfoward(ohe(X_tr[:1,:]),U,W,V,b,c)
Q,W,E,R,T=BPTT(xo,yo,U,W,V,b,c)
QQ,WW,EE,RR,TT=num_grad(xo,yo,U,W,V,b,c)
print np.sum(Q-QQ),np.sum(W-WW),np.sum(E-EE),np.sum(R-RR),np.sum(T-TT)




'''
#start learning
nep=1000
percent=nep/100
lr=0.001

#For adam optimizer
b1=0.99
b2=0.999
mU=np.zeros_like(U)
vU=np.zeros_like(U)
mW=np.zeros_like(W)
vW=np.zeros_like(W)
mV=np.zeros_like(V)
vV=np.zeros_like(V)
mb=np.zeros_like(b)
vb=np.zeros_like(b)
mc=np.zeros_like(c)
vc=np.zeros_like(c)

tcst=[]
vcst=[]
nbat=10
nval=int(0.1*Ndat)
datpb=(Ndat-nval+1)/nbat
index=np.arange(Ndat)
for epoch in range(nep+1):
	np.random.shuffle(index)

	#Check error function
	y,_,_=feedfoward(X_tr[index[nval:],:,:],U,W,V,b,c)
	tcst.append(cost_ms(y,Y_tr[index[nval:],:]))
	yv,_,_=feedfoward(X_tr[index[:nval],:,:],U,W,V,b,c)
	vcst.append(cost_ms(yv,Y_tr[index[:nval],:]))
	
	if epoch%percent is 0:
		print epoch, tcst[epoch],vcst[epoch]#,U,W,V,b,c#,gU,gW,gV,gb,gc,y[0,0],Y_tr[0,0]

	#stochastic gradient
	for batch in range(nbat):
		bst=batch*datpb

		#Get gradient : Backpropagation through time
		gU,gW,gV,gb,gc=BPTT(X_tr[index[nval+bst:nval+bst+datpb],:,:],Y_tr[index[nval+bst:nval+bst+datpb],:],U,W,V,b,c)
	
	
		#Adam optimizer
		mU=b1*mU+(1-b1)*gU
		vU=b2*vU+(1-b2)*np.square(gU)	
		mU_h=mU/(1-b1**(epoch+1))
		vU_h=vU/(1-b2**(epoch+1))
	
		mW=b1*mW+(1-b1)*gW
		vW=b2*vW+(1-b2)*np.square(gW)	
		mW_h=mW/(1-b1**(epoch+1))
		vW_h=vW/(1-b2**(epoch+1))
	
		mV=b1*mV+(1-b1)*gV
		vV=b2*vV+(1-b2)*np.square(gV)	
		mV_h=mV/(1-b1**(epoch+1))
		vV_h=vV/(1-b2**(epoch+1))
	
		mb=b1*mb+(1-b1)*gb
		vb=b2*vb+(1-b2)*np.square(gb)	
		mb_h=mb/(1-b1**(epoch+1))
		vb_h=vb/(1-b2**(epoch+1))
	
		mc=b1*mc+(1-b1)*gc
		vc=b2*vc+(1-b2)*np.square(gc)	
		mc_h=mc/(1-b1**(epoch+1))
		vc_h=vc/(1-b2**(epoch+1))
	
		#update
		U-=lr*np.divide(mU_h,np.sqrt(vU_h)+1e-8)
		W-=lr*np.divide(mW_h,np.sqrt(vW_h)+1e-8)
		V-=lr*np.divide(mV_h,np.sqrt(vV_h)+1e-8)
		b-=lr*np.divide(mb_h,np.sqrt(vb_h)+1e-8)
		c-=lr*np.divide(mc_h,np.sqrt(vc_h)+1e-8)

#Show training curve
plt.xlabel('epoch')
plt.ylabel('loss')
plt.yscale('log')
plt.plot(np.arange(len(vcst)),vcst,lw=1,label='validation')
plt.plot(np.arange(len(tcst)),tcst,lw=1,label='test')
plt.legend()
plt.savefig('RNN_lc.png',bbox_inches='tight',dpi=300)
plt.show()

#Test
Ndat=1
Y_ts=X_tr[0,:,:].reshape(tau)
for i in np.arange(0,tlen-tau):
	X_ts=Y_ts[i:i+tau].reshape(1,tau,idim)
	y,_,_=feedfoward(X_ts,U,W,V,b,c)
	Y_ts=np.append(Y_ts,y)


#Show predicted data
plt.xlabel('x')
plt.ylabel('y')
plt.plot(np.arange(tlen)/(4*np.pi),sindat,lw=1,label='original')
plt.plot(np.arange(tlen)/(4*np.pi),Y_ts,lw=1,label='prediction')
plt.plot(np.arange(tlen)/(4*np.pi),np.sin(np.arange(0,4.*np.pi,4.*np.pi/tlen)),lw=1,label='clean')
plt.legend()
plt.savefig('RNN_sin.png',bbox_inches='tight',dpi=300)
plt.show()
'''
