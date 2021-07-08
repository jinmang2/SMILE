import pickle

import numpy as np

import torch
import torch.nn as nn

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertConfig
from tensorflow.keras.preprocessing.sequence import pad_sequences


def convert_input_data(sentences):
    # BERT의 토크나이저로 문장을 토큰으로 분리
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    # 입력 토큰의 최대 시퀀스 길이
    MAX_LEN = 128

    # 토큰을 숫자 인덱스로 변환
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # 어텐션 마스크 초기화
    attention_masks = []

    # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
    # 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    # 데이터를 파이토치의 텐서로 변환
    inputs = torch.LongTensor(input_ids)
    masks = torch.LongTensor(attention_masks)

    return inputs, masks


def test_sentences(sentences, model, device):

    # 평가모드로 변경
    model.eval()

    # 문장을 입력 데이터로 변환
    inputs, masks = convert_input_data(sentences)

    # 데이터를 GPU에 넣음
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)

    # 그래디언트 계산 안함
    with torch.no_grad():
        # Forward 수행
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask)

    # 로스 구함
    logits = outputs[0]

    # CPU로 데이터 이동
    logits = logits.detach().cpu().numpy()

    return logits


def predict(inputs, model, device):
    sentences = inputs['texts'].values()
    logits = test_sentences(sentences, model, device)
    arrs = np.exp(logits)
    arrs = arrs / arrs.sum(axis=1).reshape(-1, 1)
    return {
        id:{'긍정':arr[1], '부정':arr[0]}
        for id, arr in zip(inputs['texts'].keys(), arrs)
    }


with open('bertconfig200724.pkl', 'rb') as f:
    config = pickle.load(f)

config.num_labels = 2

GPU_NUM = 0
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-multilingual-cased', do_lower_case=False)
model = BertForSequenceClassification(config)
model.load_state_dict(torch.load('bert200724.pt'))
model.to(device)
