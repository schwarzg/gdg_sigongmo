import numpy as np  # Array 관련 함수

### 한 노래에 대한 코드 단어장을 만드는 함수
def Make_vocabuluary(seq):  
    voca_codes = np.unique(seq)  # 유니크한 노드만 저장 (한 노래의 전체 음표 단어장)
    code2idx = dict([(c, i) for i, c in enumerate(voca_codes)])  # 코드를 숫자로 변환
    idx2code = dict([(i, c) for i, c in enumerate(voca_codes)])  # 숫자를 코드로 변환

    len_voca = len(voca_codes)  # 단어장 크기

    print('노래 코드의 길이는 ', len(seq), ' 입니다.')
    print('노래 코드의 유니크한 코드 수는 ', len_voca, '입니다.')
    return code2idx, idx2code, len_voca


### 지정된 크기(window size)만큼 잘라서 데이터 셋을 생성하는 함수
def Seq2data(seq, window_size, code2idx):  # window_size = 데이터 input 수 + output 수
    dataset = []
    for i in range(len(seq) - window_size):  # (전체 길이 – window 길이+1) 값=dataset의 데이터 수
        subset = seq[i:(i + window_size)]  # window 길이만큼의 데이터를 삽입
        dataset.append([code2idx[item] for item in subset]) # 코드형태의 값을 정수값으로 저장 ex> C4_0.5' --> 2

    dataset = np.array(dataset) # list 형태를 array 형태로 변환
    print('데이터 셋의 모양(shape)는', dataset.shape, '입니다')
    return dataset


### 입력(X)와 출력(Y)의 데이터 셋 만드는 함수
def Separate_X_and_Y(data_set, input_size, output_size):
    ### 입력 및 출력 데이터 셋 얻기
    x_train = data_set[:, :input_size]  # 입력 데이터 셋
    y_train = data_set[:, input_size]  # 출력 데이터 셋

    ### 입력값 정규화
    x_train = x_train / float(output_size) # 정수값을 0~1사이의 값으로 정규화
    x_train = x_train.reshape(len(x_train), input_size, 1)  # (a,b) to (a,b,1)

    ### 출력값 벡터로 표현 (one-hot encoding): 인덱스로 표현된 단어를 벡터로 표현
    y_train = np.eye(output_size)[y_train]  # ex> 2 --> [0,0,1,0,0, ..., 0]

    return x_train, y_train
