### 모듈 불러오기
import numpy as np  # Array 관련 함수
from music21.note import Note, Rest  # 음악 관련 함수
from music21.tie import Tie
from music21.stream import Stream
from music21.converter import parse
from music21.instrument import partitionByInstrument
from music21.chord import Chord

### 음악 정보가 정의된 함수
def Music_code(no):
    music_title = ['나비야', '들장미소녀캔디', '사랑을했다', '봄봄봄']
    # seq_v0, seq_v1, seq_v2, seq_v3가 정의되어 있음
    ## 나비야
    seq_v0 = [
        'G4_0.5', 'E4_0.5', 'E4_1', 'F4_0.5', 'D4_0.5', 'D4_1',  # 나비야 나비야
        'C4_0.5', 'D4_0.5', 'E4_0.5', 'F4_0.5', 'G4_0.5', 'G4_0.5', 'G4_1',  # 이리날아 오너라
        'G4_0.5', 'E4_0.5', 'E4_0.5', 'E4_0.5', 'F4_0.5', 'D4_0.5', 'D4_1',  # 노랑나비 흰나비
        'C4_0.5', 'E4_0.5', 'G4_0.5', 'G4_0.5', 'E4_0.5', 'E4_0.5', 'E4_1',  # 춤을 추며 오너라
        'D4_0.5', 'D4_0.5', 'D4_0.5', 'D4_0.5', 'D4_0.5', 'E4_0.5', 'F4_1',  # 봄바람에 꽃잎도
        'E4_0.5', 'E4_0.5', 'E4_0.5', 'E4_0.5', 'E4_0.5', 'F4_0.5', 'G4_1',  # 방긋방긋 웃으며
        'G4_0.5', 'E4_0.5', 'E4_1', 'F4_0.5', 'D4_0.5', 'D4_1',  # 참새도 짹짹짹
        'C4_0.5', 'E4_0.5', 'G4_0.5', 'G4_0.5', 'E4_0.5', 'E4_0.5', 'E4_1']  # 노래하며 춤춘다

    ## 들장미 소녀 캔디
    seq_v1 = [
        'E4_0.5', 'F4_0.5', 'G4_0.5', 'G4_0.5', 'G4_1', 'A4_0.5', 'E4_0.5', 'G4_1', 'RE_1', 'RE_2',  # 외로워도 - 슬퍼도
        'C5_1', 'B4_1', 'A4_1', 'G4_1', 'F4_4',  # 나는 안울어
        'D4_0.5', 'E4_0.5', 'F4_1', 'F4_1.5', 'F4_0.5', 'F4_2', 'B4_1', 'A4_1',  # 참-고 참고 또 참지
        'G4_1', 'G4_1', 'F4_1', 'G4_1', 'E4_4',  # 울긴 왜 울어
        'E4_0.5', 'E4_0.5', 'E4_0.5', 'F4_0.5', 'G4_0.5', 'G4_0.5', 'G4_0.5', 'G4_0.5',  # 웃으면서 달려보자
        'C5_1', 'B4_1', 'A4_1', 'G4_1',  # 푸른들을
        'D4_0.5', 'D4_0.5', 'D4_0.5', 'E4_0.5', 'F4_0.5', 'F4_0.5', 'F4_0.5', 'F4_0.5',  # 푸른하늘 바라보며
        'B4_1', 'A4_1', 'G4_1', 'F4_1',  # 노래하자
        'RE_0.5', 'E4_0.5', 'E4_0.5', 'F4_0.5', 'G4_2',  # 내이름은
        'RE_0.5', 'A4_0.5', 'A4_0.5', 'B4_0.5', 'C5_2',  # 내 이름은
        'B4_0.5', 'B4_0.5', 'B4_0.5', 'C5_0.5', 'D5_2', 'C5_1', 'RE_1', 'RE_2']  # 내 이름은 캔디

    ## 아이콘 - 사랑을 했다
    seq_v2 = [
        'RE_2', 'RE_0.5', 'B4_0.5', 'A4_0.5', 'G4_0.5',  # 사랑을
        'B4_0.5', 'D5_1.5', 'RE_0.5', 'B4_0.5', 'A4_0.5', 'G4_0.5',  # 했다 우리가
        'B4_0.5', 'A4_1.5', 'RE_0.5', 'B4_0.5', 'A4_0.5', 'G4_0.5',  # 만나 지우지
        'A4_0.5', 'A4_1', 'G4_0.5', 'A4_0.5', 'G4_0.5', 'B4_0.5', 'G4_0.5',  # 못할 추억이 됐다
        'G4_1.5', 'B4_0.5', 'A4_0.5', 'G4_0.5', 'A4_0.5', 'G4_0.5',  # 볼만한 멜로
        'B4_0.5', 'D5_1.5', 'RE_0.5', 'B4_0.5', 'A4_0.5', 'G4_0.5',  # 드A마 괜찮은
        'B4_0.5', 'A4_1.5', 'RE_0.5', 'B4_0.5', 'A4_0.5', 'G4_0.5',  # 결말 그거면
        'A4_0.5', 'A4_1', 'G4_0.5', 'A4_0.5', 'G4_0.5', 'B4_0.5', 'G4_0.5',  # 됐다 널 사랑했다
        'G4_1.5', 'B4_0.5', 'A4_0.5', 'G4_0.5', 'G4_0.5', 'G4_0.5',  # 우리가 만든
        'E5_1', 'D5_0.5', 'B4_0.5', 'B4_0.5', 'A4_0.5', 'G4_0.5',  # lo-ve 시나
        'B4_0.5', 'A4_1.5', 'RE_0.5', 'G4_0.5', 'G4_0.5', 'G4_0.5',  # 리오 이젠조
        'E5_1', 'D5_0.5', 'B4_0.5', 'B4_1', 'A4_0.5', 'G4_0.5',  # 명-----이꺼
        'A4_0.5', 'G4_1.5', 'RE_0.5', 'G4_0.5', 'G4_0.5', 'G4_0.5',  # 지고 마지막
        'E5_1', 'G5_0.5', 'E5_0.5', 'E5_1', 'E5_0.5', 'G5_0.5',  # 페이지-를 넘
        'E5_0.5', 'D5_1.5', 'RE_0.5', 'G4_0.5', 'G4_0.5', 'G4_0.5',  # 기면 조용히
        'E5_1', 'D5_0.5', 'B4_0.5', 'B4_1', 'A4_0.5', 'G4_0.5',  # 막----을 내
        'A4_0.5', 'G4_1.5', 'RE_1', 'B4_1']  # 리죠 에이

    ### 로이킴 - 봄봄봄
    seq_v3 = [
        'B4_0.75', 'B4_0.25', 'TI_-1', 'B4_0.25', 'B4_0.75',  # _ 봄봄-봄
        'B4_0.5', 'C5#_0.25', 'B4_0.25', 'TI_-1', 'B4_0.25', 'A4_0.5', 'G4#_0.25', 'TI_-1',  # _ 봄이왔-네요
        'G4#_0.75', 'C4#_0.25', 'E4_0.25', 'F4#_0.5', 'G4#_0.25', 'TI_-1',  # _ -우리가처
        'G4#_0.25', 'F4#_0.5', 'E4_0.25', 'TI_-1',  # _ -음만
        'E4_0.25', 'B3_0.25', 'C4#_0.25', 'B3_0.25', 'TI_-1',  # _ -났-던
        'B3_0.75', 'B3_0.25', 'E4_0.5', 'E4_0.25', 'G4#_0.25',  # _ -그때의향
        'G4#_0.25', 'F4#_0.75', 'E4_0.25', 'F4#_0.5', 'G4#_0.25', 'TI_-1',  # _ -기그대로
        'G4#_1', 'RE_1', 'RE_1', 'RE_0.75', 'B3_0.25',  # _ 그
        'B4_1.5', 'B4_0.5',  # _ 대가
        'B4_0.5', 'C5#_0.25', 'B4_0.25', 'TI_-1', 'B4_0.25', 'A4_0.5', 'G4#_0.25', 'TI_-1',  # _ 앉아있-었던
        'G4#_0.75', 'E4_0.25', 'E4_0.5', 'F4#_0.5',  # _ -그벤치
        'G4#_0.5', 'F4#_0.5', 'E4_0.25', 'C4#_0.5', 'E4_0.25', 'TI_-1',  # _ 옆에나무도
        'E4_0.75', 'B3_0.25', 'E4_0.5', 'E4_0.5',  # _ -아직도
        'G4#_0.25', 'F4#_0.75', 'E4_0.25', 'F4#_0.5', 'E4_0.25', 'TI_-1',  # _ 남아있네요
        'E4_1', 'RE_1', 'RE_2']

    if no == 0:
        seq_in = seq_v0  # 나비야
    elif no == 1:
        seq_in = seq_v1  # 들장미 소녀 캔디
    elif no == 2:
        seq_in = seq_v2  # 아이콘 - 사랑을 했다
    else:
        seq_in = seq_v3  # 로이킴 - 봄봄봄

    ### 선택된 곡을 저장하기
    input_fname = str('input_') + music_title[no] + str('.mid')  # 선택된 곡의 입력 파일 이름 저장
    output_fname = str('output_') + music_title[no] + str('.mid')  # 선택된 곡의 출력 파일 이름 저장
    print('선택된 노래는', music_title[no], '입니다.')
    return input_fname, output_fname, seq_in

### 전체 악보 단어장 만드는 함수
def Make_total_vocabuluary():  ### 모든 코드에 대한 코드 단어장
    ### 음표 사전
    voca_octave = np.array([3, 4, 5])  # 옥타브(기본: 4)
    voca_note = ['C', 'D', 'E', 'F', 'G', 'A', 'B']  # 도, 레, 미, 파, 솔, 라, 시
    voca_note_len = [1 / 4, 1 / 2, 3 / 4, 1, 3 / 2, 2, 3, 4]  # 16분, 8분, 점8분, 4분, 점4분, 2분, 점2분, 온 음표
    voca_accidentals = ['#', '_']  # 변화표(accidentals) 올림표(#), 내림표(-), nothing(_)
    rest = 'RE'  # 쉼표
    tie = 'TI_-1'  # 붙임줄
    voca_codes = []  # 전체 단어장
    for item_oct in voca_octave:  # (3x7x8x3)개 (옥타브 수 x 음표 수 x 박자 수 x 변화표 수)
        for item_note in voca_note:
            for item_accidentals in voca_accidentals:
                for item_note_len in voca_note_len:
                    one_item = item_note + str(item_oct) + item_accidentals + str(item_note_len)
                    voca_codes.append(one_item)  # 음악 코드 예: C4_1

    for item_note_len in voca_note_len:  # 쉼표
        one_item = rest + '_' + str(item_note_len)  # 코드 예: RE_1
        voca_codes.append(one_item)

    voca_codes.append(tie)  # 붙임줄

    code2idx = dict([(c, i) for i, c in enumerate(voca_codes)])  # 코드를 숫자로 변환
    idx2code = dict([(i, c) for i, c in enumerate(voca_codes)])  # 숫자를 코드로 변환
    print('단어장의 길이는 ', len(voca_codes), '입니다.')
    return code2idx, idx2code




### 악보 및 음원 파일을 만드는 클래스
class MAKE_SCORE:
    ### 초기화 함수
    def __init__(self, seq, fname):
        self.seq = seq
        self.fname = fname
        self.fname_xml = fname.replace('.mid', '.xml')

    ### 음표(note)인지 화음(chord)인지 확인하는 함수
    def Check_note_chord(self, element, element_len, score):
        if ('.' in element) or element.isdigit():  # 화음 일 때 ex> C4.E4.G4
            notes_in_chord = element.split('.')  # 하나의 화음 안에 노트 분리 ex> C4.E4.G4 --> ['C4', 'E4', 'G4']
            note = Chord(notes_in_chord)  # 하나의 화음을 note에 저장
            note.quarterLength = element_len  # 하나의 화음에 박자 설정

        else:  # note일 때
            note = Note(element)  # 하나의 음을 note에 저장 ex> C4
            note.quarterLength = element_len  # 하나의 음에 박자 설정

        score.append(note)  # 만든 화음이나 노트를 악보에 넣기
        return score

    ### 음악 시퀀스로부터 악보를 만드는 함수
    def Make_score(self):
        score = Stream()
        notes = self.seq[0]  # 하나의 음표 또는 화음
        notes_len = [float(item) for item in self.seq[1]]  # 하나의 음표나 화음의 박자
        n_tie = 0  # tie 수
        tie_gap = 2  # 코드 사이의 tie gap(기본값=2) ex> ['C4_0.5', 'TI_-1', 'C4_0.5']일 경우 tie_gap = 2
        i = 0
        while i < len(notes):
            if notes[i] == 'TI':  # 현재 데이터가 붙임줄(tie)일 경우 (1번 if문)
                i = i + 1  # 현재 위치를 한칸 이동
                n_tie = n_tie + 1  # tie 수 증가
                # 다음 데이터가 없는지 (2번 if문), 다른 데이터(3번 if문)가 나왔는지 확인
                if i >= len(notes):  # 다음 노트가 없을 경우 while문 종료 (2번 if문)
                    break
                if notes[i] == 'TI':  # 또 Tie가 나왔을 경우 (3번 if문)
                    tie_gap = tie_gap + 1 # tie_gap만 +1 만 해주고 (1번 if문)으로 돌아감
                    continue
                elif notes[i] == 'RE':  # Rest이 나왔을 경우
                    score.append(Rest(quarterLength=notes_len[i])) # 쉼표를 악보에 삽입
                else:  # 음이나 화음일 경우
                    score = self.Check_note_chord(notes[i], notes_len[i], score) # 음이나 화음을 삽입
                if notes[i - tie_gap] == notes[i]:  # 두 개의 음이 같은 레벨의 음일 경우(도-도 o, 도-레 x) (4번 if문)
                    score[i - n_tie - 1].tie = Tie('start')  # 붙임줄 삽입
                    tie_gap = 2 # 붙임줄을 삽입했기 때문에 다시 기본값으로 돌아감

            elif notes[i] == 'RE':  # 현재 데이터가 쉼표일 경우, 쉼표와 그의 길이를 악보스트림에 삽입
                score.append(Rest(quarterLength=notes_len[i]))
            else:  # 현재 데이터가 음이나 화음일 경우, 이를  박자와 함께 악보 스트림에 삽입
                score = self.Check_note_chord(notes[i], notes_len[i], score)
            i = i + 1

        score.write('midi', fp=self.fname)  # .mid 파일로 저장
        score.write('xml', fp=self.fname_xml)  # .xml 파일로 저장

        return score


# 노래의 음 정보와 박자 데이터 추출하는 함수
def Separate_note_and_length(seq):
    notes = []  # 계이름 및 옥타브
    notes_len = []  # 길이: 4분 음표, 8분 음표...

    for element in seq: # seq에 들어 있는 데이터 하나씩 가지고 오기
        note_and_length = element.split('_')  # (음 또는 화음)과 박자 분리   ex> G4_0.5 --> 'G4', '0.5'
        notes.append(note_and_length[0])  # 노트 시퀀스에 하나의 (음 또는 화음) 삽입 ex> 'G4'
        if ('/' in note_and_length[1]):  # 박자가 문자열 분수로 표현되어 있을 경우 ex> '1/6'
            num, denom = note_and_length[1].split('/')  # 분자와 분모 추출 ex> num='1', denom='6'
            frac = float(num) / float(denom)  # 분수형태에서 실수(float)형태로 저장 ex> 0.16666
            notes_len.append(str(frac))  # 실수 형태의 박자를 문자열로 저장  ex> 0.16666 --> '0.16666'
        else: # 박자가 문자열 실수로 표현되어 있을 경우
            notes_len.append(note_and_length[1])  # 박자 저장 ex> '0.5'

    seq_notes_and_len = np.vstack((notes, notes_len)) # note와 note 박자를 하나의 배열로 만들기
    # seq_notes_and_len[0] = notes, seq_notes_and_len[1] = notes_len
    return seq_notes_and_len


### 제공받은 음원으로 입력/출력 파일 이름을 만드는 함수
def Make_fname(path):
    folder_name, fname = path.split('/')  # folder 이름: music_collection, fname: 선택한 음원명
    midi = parse(path)  # parse: 파일을 파싱하고 스트림에 넣어주기
    input_fname = str('input_') + fname  # 시퀀스 입력 파일 이름
    output_fname = str('output_') + fname  # 시퀀스 출력 파일 이름
    music_title = fname.replace(".mid", "")  # 제목 추출
    music_title = music_title.replace("_", " ")
    print('선택된 노래는', music_title, '입니다.')
    return input_fname, output_fname, midi


### 제공받은 음악을 통해서 노트 시퀀스 만들기
def Make_notes(midi):
    parts = partitionByInstrument(midi)  ### 악기 수, 멀티 파트일 경우 각각 악기별로 파이션을 나눔

    if parts:  # 파일이 제공하는 악기가 여러 개 있을 경우
        notes_to_parse = parts.parts[0].recurse()
    else:  # 파일이 제공하는 악기가 없고 flat일 경우
        notes_to_parse = midi.flat.notes
    seq = []

    for element in notes_to_parse:
        if isinstance(element, Note):  # 노트일 경우, one_note에 음과 박자 저장 ex> 'G4_0.5'
            one_note = str(element.pitch) + '_' + str(element.duration.quarterLength)
            seq.append(one_note)  # 하나의 음를 악보에 삽입
        elif isinstance(element, Chord):  # Chord 일 경우, one_note에 화음과 박자 저장 ex> 'C4.E4.G4_0.5'
            one_note = '.'.join(str(n) for n in element.pitches) + '_' + str(element.duration.quarterLength)
            seq.append(one_note)  # 하나의 화음을 악보에 삽입
        elif isinstance(element, Rest):  # 쉼표일 경우 ex> 'RE_1.0'
            one_note = 'RE_' + str(element.duration.quarterLength)
            seq.append(one_note)   # 하나의 쉼표를 악보에 삽입

    seq = np.array(seq)  # list 형태를 np.array로 변환
    return seq

