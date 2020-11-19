import re
import pickle

import torch
import numpy as np
from chatspace import ChatSpace

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from konlpy.tag import Okt


okt = Okt()
def tokenizer_morphs(doc):
    return okt.morphs(doc)


spacer = ChatSpace()

spacer = ChatSpace(device= torch.device('cuda:0'))

i2senti = {
    0 : 'joy',
    1 : 'interest',
    2 : 'anger',
    3 : 'admiration',
    4 : 'sadness',
    5 : 'surprise',
    6 : 'fear',
    7 : 'disgust'
}


def predict(text, embedding, model):
    t = embedding.transform(
        [
            spacer.space(
                re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣]', ' ', text)
            )
        ]
    )
    d = model.predict_proba(t)

    return ' '.join(
        [i2senti[i] + f'/{d[0][i]:.2%}' for i in np.argsort(d)[0][:4:-1]]
    )

with open('gnb_clf_tfidf_20191112.pkl', 'rb') as f:
    jst_mb_model = pickle.load(f)

with open('embedding_20191112.pkl', 'rb') as f:
    embedding = pickle.load(f)

embedding.tokenizer = tokenizer_morphs
