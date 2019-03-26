### Step 1. 모듈 가지고 오기
import numpy as np  # Array 관련 함수
import about_music as AM # 음악과 관련된 라이브러리 호출
import about_data as AD # 데이터 처리와 관련된 라이브러리 호출
from tensorflow.python import  keras as K  # Tensorflow 안의 keras 관련 함수

######## A: 만든 노래 데이터들를 통해서 노래 선정하기
######## B: 제공받은 음원 파일(mid)을 통해서 노래 선정하기
### Step 2.A. 학습할 노래 선정하기
input_fname, output_fname, slected_seq = AM.Music_code(0)  # 0: 나비야, 1: 들장미소녀캔디, 2: 사랑을 했다

#### Step 2.B. 제공되는 음원 파일 로드 및 시퀀스 데이터로 만들기
# # 제공되는 음원 파일 로드하기
# fname = 'music_collection/dean-what2do.mid' # 'music_collection/twice-knock_knock.mid'
# input_fname, output_fname, midi = AM.Make_fname(fname)
# # 학습할 음악을 데이터 만들기
# slected_seq = AM.Make_notes(midi)

### Step 3. 단어장 만들기
code2idx, idx2code, output_size = AD.Make_vocabuluary(slected_seq)

### Step 4. 데이터 셋 생성하기
window_size = 5
input_size = window_size - 1
data_set = AD.Seq2data(slected_seq, window_size, code2idx)  # data_set의 값들은 int 값

### Step 5. 입력과 출력 데이터 셋 추출하기
x_train, y_train, = AD.Separate_X_and_Y(data_set, input_size, output_size)

'''  코드를 작성하세요  '''
### Step 6. 모델 만들기
model = K.Sequential()
model.add(K.layers.LSTM(128, batch_input_shape=(1,input_size, 1), stateful=True))
                        # 상태유지(Stateful) : 현재 샘플의 학습 상태가 다음 샘플의 초기 상태로 전달됨
                        # batch_input_shape = (배치사이즈, 타임스텝, 속성)
model.add(K.layers.Dense(output_size, activation='softmax'))

# 모델 학습 설정하기
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

### Step 7. 학습하기
epochs = 10
for epoch_idx in range(epochs):
    print('epochs : ' + str(epoch_idx))
    model.fit(x_train, y_train, epochs=1, batch_size=1,verbose=2, shuffle=False)
    model.reset_states()  # 상태 유지 reset


### Step 8. 예측하기
# 예측에 들어갈 시퀀스(seq_in) 초기 배열 만들기
seq_in = slected_seq[0:input_size] # seq_in에는 코드값이 저장 ['G4_0.5', 'E4_0.5', 'E4_1', 'F4_0.5']
seq_in = [code2idx[item] / float(output_size) for item in seq_in] # 코드를 정수로 변경 & 정규화 ex> [0.77,0.33,0.44,0.55]

# 예측된 시퀀스(seq_out) 초기 배열 만들기
predicted_seq = [item for item in slected_seq[0:input_size]] # seq(list)에 있는 하나의 item씩 가지고 와서 배열(array)에 저장

# 노래 예측하기
predict_size = len(slected_seq) - input_size  # 최대 예측 수 정의
for i in range(predict_size):
    predicted_in = np.array(seq_in)  # list형태에서 np.array형태로 저장
    predicted_in = predicted_in.reshape(1, len(predicted_in), 1)  # LSTM에 넣기 위해서 변수 shape 변경 (4,) --> (1,4,1)
    predicted_out = model.predict(predicted_in)  # 예측으로 하나의 벡터값이 나옴 ex> [0.09,0.14,0.10,0.20,0.09,...,0.09]

    # One-hot decoding
    predicted_idx = np.argmax(predicted_out)  # vector를 하나의 int값으로 변환 ex> [0.09,0.14,0.10,0.20,0.09,...,0.09]-->3
    predicted_code = idx2code[predicted_idx]  # int를 code로 변환 # 3 --> F4_0.5
    predicted_seq.append(predicted_code)  # 예측된 출력값을 배열에 code로 저장

    seq_in.append(predicted_idx / float(output_size))  # 예측된 idx값을 정규화 및 배열에 삽입
    seq_in.pop(0)  # seq_in에 가장 앞에 저장된 데이터를 삭제

'''  코드를 작성 끝  '''

### Step 9. 선택/학습된 노래에 음과 박자 데이터 추출하기
slected_seq_notes_and_notelen = AM.Separate_note_and_length(slected_seq)  # 선택된 노래
predicted_seq_notes_and_notelen= AM.Separate_note_and_length(predicted_seq)  # 학습된 노래

### Step10. 선택된 노래와 학습된 노래 음원 파일과 악보 저장하기
selected_score = AM.MAKE_SCORE(slected_seq_notes_and_notelen, input_fname)  # 선택된 노래
selected_score.Make_score() # 선택된 악보 작곡 및 음악 파일 저장
predicted_score = AM.MAKE_SCORE(predicted_seq_notes_and_notelen, output_fname)  # 학습된 노래
predicted_score.Make_score() # 학습된 악보 작곡 및 음악 파일 저장
