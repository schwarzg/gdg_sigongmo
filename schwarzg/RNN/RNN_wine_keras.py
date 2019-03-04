import numpy as np
import pandas as pd
import keras
import csv
import itertools
import nltk
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD,Adam
from keras.layers.recurrent import SimpleRNN,LSTM

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
sentences=nltk.sent_tokenize(train_string)[:50]

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
idim=len(word_to_index)
itau=20
htau=itau
otau=1
print idim
x_tr=[]
y_tr=[]
for i in range(0,tlen-itau):
	x_tr.append(X_tr[i:i+itau])
	y_tr.append(X_tr[i+itau+1-otau:i+itau+1])
X_tr=np.array(x_tr).reshape(len(x_tr),itau)
Y_tr=np.array(y_tr).reshape(len(y_tr),otau)
Ndat=len(X_tr)

print X_tr.shape
print Y_tr.shape

def ohe(x,vsize):
	ohev=np.zeros((len(x),len(x[0]),vsize))
	for i in range(len(x)):
		for j in range(len(x[0])):
			ohev[i,j,x[i,j]]=1
	return ohev

X_tr=ohe(X_tr,idim)
Y_tr=ohe(Y_tr,idim)
Y_tr=Y_tr[:,0,:]

telen=int(Ndat*0.1)
X_val=X_tr[:telen,:,:]
Y_val=Y_tr[:telen,:]

X_tr=X_tr[telen:,:,:]
Y_tr=Y_tr[telen:,:]

hdim=idim*2

model=Sequential()
model.add(SimpleRNN(hdim,init='lecun_normal',input_shape=(itau,idim)))
model.add(Dense(idim,init='lecun_normal'))
model.add(Activation('softmax'))

model.compile(loss='kullback_leibler_divergence',optimizer=Adam(lr=0.001,beta_1=0.9,beta_2=0.999))

model.fit(X_tr,Y_tr,
	batch_size=10,
	epochs=500,
	validation_data=(X_val,Y_val))


