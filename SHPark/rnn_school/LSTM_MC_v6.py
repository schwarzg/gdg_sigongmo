#######################################################################################################
###                                            메인 코드                                            ###
#######################################################################################################
### Step 1. 모듈 가지고 오기
import numpy as np  # Array 관련 함수
import about_music as AM # 음악과 관련된 라이브러리 호출
import about_data as AD # 데이터 처리와 관련된 라이브러리 호출
from tensorflow.python import  keras as K  # Tensorflow 안의 keras 관련 함수

### Step 2.A. 학습할 노래 선정하기
input_fname, output_fname, seq = AM.Music_code(0)  # 0: 나비야, 1: 들장미소녀캔디, 2: 사랑을 했다, 3: 봄봄봄

### Step 3. 단어장 만들기
code2idx, idx2code, output_size = AD.Make_vocabuluary(seq)

### Step 4. 데이터 셋 생성하기
window_size = 5
input_size = window_size - 1
data_set = AD.Seq2data(seq, window_size, code2idx)  # data_set의 값들은 int 값

### Step 5. 입력과 출력 데이터 셋 만들기
x_train, y_train, = AD.Separate_X_and_Y(data_set, input_size, output_size)

'''  코드를 작성하세요  '''
### Step 6. 모델 만들기
model = K.Sequential()
model.add(K.layers.LSTM(512, input_shape=(input_size, 1), return_sequences=True))
model.add(K.layers.LSTM(512, return_sequences=True))
model.add(K.layers.Dropout(0.3))
model.add(K.layers.LSTM(256))
model.add(K.layers.Dense(128))
model.add(K.layers.Dropout(0.3))
model.add(K.layers.Dense(output_size, activation='softmax'))

# 모델 학습 설정하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

### Step 7. 학습하기
epochs = 10
batch_size = 64
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=False)


### Step 8. 예측하기
# 예측에 들어갈 시퀀스(seq_in) 초기 배열 만들기
seq_in = seq[0:input_size] # seq_in에는 코드값이 저장 ['G4_0.5', 'E4_0.5', 'E4_1']
seq_in = [code2idx[item] / float(output_size) for item in seq_in] # 코드를 정수로 변경 & 정규화 ex> [0,1, ...]

# 예측된 시퀀스(seq_out) 초기 배열 만들기
seq_out = [item for item in seq[0:input_size]] # seq(list)에 있는 하나의 item씩 가지고 와서 배열(array)에 저장

# 노래 예측하기
predict_size = len(seq) - input_size  # 최대 예측 수 정의
for i in range(predict_size):
    predicted_in = np.array(seq_in)  # list형태에서 np.array형태로 저장
    predicted_in = predicted_in.reshape(1, len(predicted_in), 1)  # LSTM에 넣기 위해서 변수 shape 변경 (4,) --> (1,4,1)
    predicted_out = model.predict(predicted_in)  # 예측으로 하나의 벡터값이 나옴 ex> [0,0,1,0,0,...,0]

    # One-hot decoding
    predicted_idx = np.argmax(predicted_out)  # vector를 하나의 int값으로 변환 ex> [0,0,1,0,0,...,0] --> 2
    predicted_code = idx2code[predicted_idx]  # int를 code로 변환 # 2 --> D4_0.5
    seq_out.append(predicted_code)  # 예측된 출력값을 배열에 code로 저장

    seq_in.append(predicted_idx / float(output_size))  # 예측된 idx값을 정규화 및 배열에 삽입
    seq_in.pop(0)  # seq_in에 가장 앞에 저장된 데이터를 삭제

'''  코드를 작성 끝  '''

### Step 9. 선택된 노래와 학습된 노래 음원 파일과 악보 저장하기
# 선택된 노래 음원 파일과 악보 만들기
seq_in_notes_and_notelen = AM.Separate_note_and_length(seq)  # 노래의 음 정보와 음의 박자 데이터 추출
SCORE = AM.MAKE_SCORE(seq_in_notes_and_notelen, input_fname)  # 악보 작곡에 필요한 데이터 저장
SCORE.Make_score() # 악보 작곡 및 음악 파일 저장

# 학습된 노래 음원 파일과 악보 만들기
seq_out_notes_and_notelen= AM.Separate_note_and_length(seq_out)  # 노래의 음 정보와 음의 박자 데이터 추출
SCORE = AM.MAKE_SCORE(seq_out_notes_and_notelen, output_fname)  # 악보 작곡에 필요한 데이터 저장
SCORE.Make_score() # 악보 작곡 및 음악 파일 저장



# ################### 제공되는 음악파일을 이용하기
# ### Step 1. 모듈 가지고 오기
# import numpy as np  # Array 관련 함수
# import about_music as AM # 음악과 관련된 라이브러리 호출
# import about_data as AD # 데이터 처리와 관련된 라이브러리 호출
# from tensorflow.python import  keras as K  # Tensorflow 안의 keras 관련 함수
#
# ### Step 2.B. 제공되는 음원 파일 로드 및 시퀀스 데이터로 만들기
# # 제공되는 음원 파일 로드하기
# fname = 'music_collection/dean-what2do.mid' # 'music_collection/twice-knock_knock.mid'
# input_fname, output_fname, midi = AM.Make_fname(fname)
#
# # 학습할 음악을 데이터 만들기
# seq = AM.Make_notes(midi)
#
# ### Step 3. 단어장 만들기
# code2idx, idx2code, output_size = AD.Make_vocabuluary(seq)
#
# ### Step 4. 데이터 셋 생성하기
# window_size = 5
# input_size = window_size - 1
# data_set = AD.Seq2data(seq, window_size, code2idx)  # data_set의 값들은 int 값
#
# ### Step 5. 입력과 출력 데이터 셋 만들기
# x_train, y_train, = AD.Separate_X_and_Y(data_set, input_size, output_size)
#
# '''  코드를 작성하세요  '''
# ### Step 6. 모델 만들기
# model = K.Sequential()
# model.add(K.layers.LSTM(512, input_shape=(input_size, 1), return_sequences=True))
# model.add(K.layers.LSTM(512, return_sequences=True))
# model.add(K.layers.Dropout(0.3))
# model.add(K.layers.LSTM(256))
# model.add(K.layers.Dense(128))
# model.add(K.layers.Dropout(0.3))
# model.add(K.layers.Dense(output_size, activation='softmax'))
#
# # 모델 학습 설정하기
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()
#
# ### Step 7. 학습하기
# epochs = 10
# batch_size = 64
# model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=False)
#
#
# ### Step 8. 예측하기
# # 예측에 들어갈 시퀀스(seq_in) 초기 배열 만들기
# seq_in = seq[0:input_size] # seq_in에는 코드값이 저장 ['G4_0.5', 'E4_0.5', 'E4_1']
# seq_in = [code2idx[item] / float(output_size) for item in seq_in] # 코드를 정수로 변경 & 정규화 ex> [0,1, ...]
#
# # 예측된 시퀀스(seq_out) 초기 배열 만들기
# seq_out = [item for item in seq[0:input_size]] # seq(list)에 있는 하나의 item씩 가지고 와서 배열(array)에 저장
#
# # 노래 예측하기
# predict_size = len(seq) - input_size  # 최대 예측 수 정의
# for i in range(predict_size):
#     predicted_in = np.array(seq_in)  # list형태에서 np.array형태로 저장
#     predicted_in = predicted_in.reshape(1, len(predicted_in), 1)  # LSTM에 넣기 위해서 변수 shape 변경 (4,) --> (1,4,1)
#     predicted_out = model.predict(predicted_in)  # 예측으로 하나의 벡터값이 나옴 ex> [0,0,1,0,0,...,0]
#
#     # One-hot decoding
#     predicted_idx = np.argmax(predicted_out)  # vector를 하나의 int값으로 변환 ex> [0,0,1,0,0,...,0] --> 2
#     predicted_code = idx2code[predicted_idx]  # int를 code로 변환 # 2 --> D4_0.5
#     seq_out.append(predicted_code)  # 예측된 출력값을 배열에 code로 저장
#
#     seq_in.append(predicted_idx / float(output_size))  # 예측된 idx값을 정규화 및 배열에 삽입
#     seq_in.pop(0)  # seq_in에 가장 앞에 저장된 데이터를 삭제
#
# '''  코드를 작성 끝  '''
#
# ### Step 9. 선택된 노래와 학습된 노래 음원 파일과 악보 저장하기
# # 선택된 노래 음원 파일과 악보 만들기
# seq_in_notes_and_notelen = AM.Separate_note_and_length(seq)  # 노래의 음 정보와 음의 박자 데이터 추출
# SCORE = AM.MAKE_SCORE(seq_in_notes_and_notelen, input_fname)  # 악보 작곡에 필요한 데이터 저장
# SCORE.Make_score() # 악보 작곡 및 음악 파일 저장
#
# # 학습된 노래 음원 파일과 악보 만들기
# seq_out_notes_and_notelen= AM.Separate_note_and_length(seq_out)  # 노래의 음 정보와 음의 박자 데이터 추출
# SCORE = AM.MAKE_SCORE(seq_out_notes_and_notelen, output_fname)  # 악보 작곡에 필요한 데이터 저장
# SCORE.Make_score() # 악보 작곡 및 음악 파일 저장

