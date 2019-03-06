'''
 Wine review
 author: Seunghwa Park
'''
'''
                      모듈 가져오기
'''
import pandas as pd
import numpy as np
import time
import re
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import GRU, LSTM, SimpleRNN
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

'''
                      initital values
'''
data_path = 'C:/Users/seunghwa/Desktop/wine-reviews/winemag-data-130k-v2.csv'
n_discrip = 500 # n개의 description review
sentence_start_token = "SENTENCESTART "
sentence_end_token = " SENTENCEEND"

'''
                      데이터 읽기
'''
read_data=pd.read_csv(data_path) # 데이터 읽기
# cafegories: unnammed / country / description / designation / points / price / provine / region_1 / region_2 /
#              taster_name / taster_twitter_haddle / tittle / variety / winery
print('\t\t\t\t [READ DATA] \n', read_data.head()) # 데이터 출력

'''
                   description & 단어 추출
'''
data_discrip = read_data["description"] # 읽어온 데이터로부터 description 추출
print('-----------------------------------------------------------------------')
print('\t\t\t [DISCRIPTION DATA] \n', data_discrip.head())
data_discrip2 = data_discrip[:n_discrip]
data_test = data_discrip[:n_discrip]

sentences = []
sentences2 = []
def slipt_sentences(raw): # dot(.)으로 문장을 분류
    making_sentences = re.split(r"\.\s*", raw)
    for i in range(len(making_sentences)-1):
        making_sentences[i] = sentence_start_token + making_sentences[i] + sentence_end_token

    return making_sentences

for raw in data_discrip2: # description에서 문장 추출
    if len(raw)>0:
        sentences2.append(slipt_sentences(raw))
for i, sentence in enumerate(sentences2):
    sentences2[i] = ' '.join(sent_one_discrip for sent_one_discrip in sentences2[i])
# # print(' '.join(sentence for sentence in sentences2[1]))
# print(sentences2[1])

def sentence_to_wordlist(raw): # 문장에서 단어 추출
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words

for raw in sentences2: # description에서 문장 추출
    if len(raw)>0:
        sentences.append(sentence_to_wordlist(raw))

token_count = sum(len(sentence) for sentence in sentences) # 토큰 수
flat_list_char = []
for sublist in sentences: # n개의 문장들을 한줄로 연결
    for item in sublist:
        flat_list_char.append(item)

token_unique = np.unique(flat_list_char) # 유니크한 토큰 추출
token_unique_size = len(token_unique) # 유니크한 토큰 수
print('The Number of Tokens: ', token_count)
print ("Total Unique Tokens:", token_unique_size)

'''
                   토큰 생성
'''
# 단어장 생성
voca_dict_char2int = dict([(w, i) for i, w in enumerate(token_unique)])  # C2I로 바꿔주는 단어장 (for training)
voca_dict_int2char = dict([(i, w) for i, w in enumerate(token_unique)]) # ItoC로 바꿔주는 단어장 (for printing test)
tokenized_sentences = sentences

'''
                   트레이닝 데이터 생성
'''
sentences_int=[]
for sentence in tokenized_sentences:
    sentences_int.append([voca_dict_char2int[w] for w in sentence]) # char to int list

print('Untokenized sentence: ', data_discrip[1])
print('Tokenized sentence: ', tokenized_sentences[1])
print('Tokenized sentence char to int: ', sentences_int[1])

flat_list_int = []
flat_list_int_ts = []
for i,sublist in enumerate(sentences_int): # train과 test set 분류 (train: n_discrip -1 개, test: 1개(맨 마지막 문장))
    if i < len(sentences_int)-1:
        for item in sublist:
            flat_list_int.append(item)
    else:
        for item in sublist:
            data_discrip_ts = data_discrip[i]
            tokenized_sentences_ts = tokenized_sentences[i]
            flat_list_int_ts.append(item)


# train과 test 데이터 만들기
itau=8  # 한번에 들어갈 input size
otau=1  # output size
X_tr = []   # training data X
Y_tr = []   # training data Y
X_ts = []   # test data X
token_count_tr = sum(len(sentence) for sentence in sentences[:-1]) # train 토큰 수
token_count_ts = len(flat_list_int_ts)                              # tes 토큰 수

for i in range(token_count_tr - itau): # Train data 생성
    X_tr.append(flat_list_int[i:i+itau])
    Y_tr.append(flat_list_int[i+itau])

for i in range(token_count_ts - itau): # Test data 생성
    X_ts.append(flat_list_int_ts[i:i+itau])

X_tr = np.array(X_tr).reshape(len(X_tr),itau)
Y_tr = np.array(Y_tr).reshape(len(Y_tr),otau)
X_ts = np.array(X_ts).reshape(len(X_ts),itau)

print('training size:', X_tr.shape, Y_tr.shape)

# int to vector (one-hot encoder)
def ohe(x,vsize):
	ohev=np.zeros((len(x),len(x[0]),vsize))
	for i in range(len(x)):
		for j in range(len(x[0])):
			ohev[i,j,x[i,j]]=1
	return ohev
cut_size_tr = len(X_tr)
cut_size_ts = len(X_ts)

xo=ohe(X_tr[:cut_size_tr,:],token_unique_size)
yo=ohe(Y_tr[:cut_size_tr,:],token_unique_size)
xo_ts = ohe(X_ts[:cut_size_ts,:],token_unique_size)

yo = yo.reshape(cut_size_tr, token_unique_size)

'''
                   모델 생성
'''
### 모델 생성
def weight_variable(shape, name=None):
    return np.random.normal(scale=.01, size=shape)
early_stopping = EarlyStopping(monitor='loss', patience=100, verbose=1)
n_hidden = 100

def build_RNN():
    model_RNN = Sequential()
    model_RNN.add(SimpleRNN(n_hidden, kernel_initializer=weight_variable, input_shape=(itau, token_unique_size)))
    model_RNN.add(Dense(n_hidden, kernel_initializer=weight_variable))
    model_RNN.add(Dense(token_unique_size, kernel_initializer=weight_variable))
    model_RNN.add(Activation('softmax'))

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model_RNN.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model_RNN

def build_LSTM():
    model_LSTM = Sequential()
    model_LSTM.add(LSTM(n_hidden, kernel_initializer=weight_variable, input_shape=(itau, token_unique_size)))
    model_LSTM.add(Dense(n_hidden, kernel_initializer=weight_variable))
    model_LSTM.add(Dense(token_unique_size, kernel_initializer=weight_variable))
    model_LSTM.add(Activation('softmax'))

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model_LSTM.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model_LSTM


def build_GRU():
    model_GRU = Sequential()
    model_GRU.add(GRU(n_hidden, kernel_initializer=weight_variable, input_shape=(itau, token_unique_size)))
    model_GRU.add(Dense(n_hidden, kernel_initializer=weight_variable))
    model_GRU.add(Dense(token_unique_size, kernel_initializer=weight_variable))
    model_GRU.add(Activation('softmax'))

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model_GRU.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model_GRU


model_RNN = build_RNN()
model_LSTM = build_LSTM()
model_GRU = build_GRU()

'''
                   트레이닝
'''
epochs = 1000
batch_size = 100

RNN_start_time = time.time()
hist_RNN = model_RNN.fit(xo, yo,
                 batch_size=batch_size,
                 epochs=epochs,
                 callbacks=[early_stopping])
RNN_running_time = time.time() -RNN_start_time

LSTM_start_time = time.time()
hist_LSTM = model_LSTM.fit(xo, yo,
                 batch_size=batch_size,
                 epochs=epochs,
                 callbacks=[early_stopping])
LSTM_running_time = time.time() -LSTM_start_time

GRU_start_time = time.time()
hist_GRU = model_GRU.fit(xo, yo,
                 batch_size=batch_size,
                 epochs=epochs,
                 callbacks=[early_stopping])
GRU_running_time = time.time() -GRU_start_time



'''
                   예측
'''
### 예측
Z_RNN = xo_ts[0][:itau].reshape(1, itau, token_unique_size)  # 본래 데이터의 itau만큼만 자름
Z_LSTM = xo_ts[0][:itau].reshape(1, itau, token_unique_size)  # 본래 데이터의 itau만큼만 자름
Z_GRU = xo_ts[0][:itau].reshape(1, itau, token_unique_size)  # 본래 데이터의 itau만큼만 자름

predicted_RNN = Z_RNN.reshape(itau, token_unique_size)
predicted_LSTM = Z_LSTM.reshape(itau, token_unique_size)
predicted_GRU = Z_GRU.reshape(itau, token_unique_size)

for i in range(100 - itau + 1):
    z_RNN = Z_RNN[:][i:i+itau][:]
    z_LSTM = Z_LSTM[:][i:i+itau][:]
    z_GRU = Z_GRU[:][i:i+itau][:]

    y_RNN = model_RNN.predict(z_RNN).reshape(otau, token_unique_size)
    y_LSTM = model_LSTM.predict(z_LSTM).reshape(otau, token_unique_size)
    y_GRU = model_GRU.predict(z_GRU).reshape(otau, token_unique_size)

    sequence_RNN = np.concatenate((z_RNN.reshape(1, itau, token_unique_size)[0][1:], y_RNN), axis=0).reshape(1, itau, token_unique_size)
    sequence_LSTM = np.concatenate((z_LSTM.reshape(1, itau, token_unique_size)[0][1:], y_LSTM), axis=0).reshape(1, itau, token_unique_size)
    sequence_GRU = np.concatenate((z_GRU.reshape(1, itau, token_unique_size)[0][1:], y_GRU),axis=0) .reshape(1, itau, token_unique_size)

    Z_RNN = np.append(Z_RNN, sequence_RNN, axis=0)
    Z_LSTM = np.append(Z_LSTM, sequence_LSTM, axis=0)
    Z_GRU = np.append(Z_GRU, sequence_GRU, axis=0)

    predicted_RNN = np.vstack((predicted_RNN, y_RNN))
    predicted_LSTM = np.vstack((predicted_LSTM, y_LSTM))
    predicted_GRU = np.vstack((predicted_GRU, y_GRU))

'''
                   결과
'''
# vector to int (one-hot decoder)
def ohd(i):
    return np.argmax(i)

predicted_RNN_int = [ohd(v) for v in predicted_RNN]
predicted_RNN_char = [voca_dict_int2char[i] for i in predicted_RNN_int]
predicted_LSTM_int = [ohd(v) for v in predicted_LSTM]
predicted_LSTM_char = [voca_dict_int2char[i] for i in predicted_LSTM_int]
predicted_GRU_int = [ohd(v) for v in predicted_GRU]
predicted_GRU_char = [voca_dict_int2char[i] for i in predicted_GRU_int]

print('\t\t[Predicted Sentence ]')
print('predicted with RNN: ')
print(" ".join(word for word in predicted_RNN_char))
print('predicted with LSTM: ')
print(" ".join(word for word in predicted_LSTM_char))
print('predicted with GRU: ')
print(" ".join(word for word in predicted_GRU_char))

print()
print('\t\t[ Running Time ]')
print('   # of discrip: ', n_discrip)                       # training discrip = n_discrip-1 (one for test)
print('   The Number of Tokens: ', token_count)             # total n_discrip의 token
print("   Total Unique Tokens:", token_unique_size)         # total n_discrip의 unique token
print("   itau: ", itau)                                    # 한번에 들어갈 input 길이
print('   epochs: ', epochs, ', batch size: ', batch_size)
print('\t\t[ Running Time ]')
print('\t RNN: ', RNN_running_time)
print('\t LSTM: ', LSTM_running_time)
print('\t GRU: ', GRU_running_time)

import matplotlib
# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 15}
#
# matplotlib.rc('font', **font)
line_width = 2.5
font_size = 15
fig = plt.figure()
text_title = 'RNN vs LSTM vs GRU'
fig.suptitle(text_title, fontsize=font_size)
ax1 = plt.subplot()
ax1.plot(hist_RNN.history['loss'], label='RNN loss', linewidth=line_width)
ax1.plot(hist_LSTM.history['loss'], label='LSTM loss', linewidth=line_width)
ax1.plot(hist_GRU.history['loss'], label='GRU loss', linewidth=line_width)
plt.xlabel('Epochs', fontsize=font_size)
plt.ylabel('loss', fontsize=font_size)
plt.legend()
plt.show()
