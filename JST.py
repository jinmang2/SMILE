import gc
from ksenticnet_kaist import *


ksenticnet = get_ksenticnet()

keys = list(ksenticnet.keys())
senticvals = [[float(i) for i in val[:4]] for val in  ksenticnet.values()]
sentiments = []
polarity = []
semantics = []
for key, val in ksenticnet.items():
    for i in val[4:]:
        if i in ['positive', 'negative']:
            polar_ind = val.index(i)
            sentiments.append(val[4 : polar_ind])
            polarity.append(val[polar_ind : polar_ind+2])
            semantics.append(val[polar_ind+2 :])
            break

ksenticnets = defaultdict(dict)
for key, val, senti, p, seman in zip(keys, 
                                     senticvals, 
                                     sentiments, 
                                     polarity, 
                                     semantics):
    ksenticnets[key]['sentic_value'] = val
    ksenticnets[key]['sentiment'] = senti
    ksenticnets[key]['polarity'] = p
    ksenticnets[key]['semantic'] = seman

f = lambda x : [i if i > 0 else 0 for i in x]
g = lambda x : [abs(i) if i < 0 else 0 for i in x]
scores = np.array(list(map(lambda x : f(x) + g(x), senticvals)))
scores /= scores.sum(axis=1).reshape(-1, 1)

class KSenticNet():
    keys = {j : i for i, j in  enumerate(keys)}
    scores = scores
    
    
MAX_VOCAB_SIZE = 50000

def sample_from_dirichlet(alpha):
    return np.random.dirichlet(alpha)

def sample_from_categorical(theta):
    theta = theta / np.sum(theta)
    return np.random.multinomial(1, theta).argmax()

def word_indices(wordOccurenceVec):
    for idx in wordOccurenceVec.nonzero()[0]:
        for i in range(int(wordOccurenceVec[idx])):
            yield idx


class KSenticNet():
    keys = {j : i for i, j in  enumerate(keys)}
    scores = scores
    
    
def processSingleReview(review, st, d=None, stopwords=None):
    letters_only = re.sub('[^ㄱ-하-ㅣ가-힣]', ' ', review).strip()
    if not stopwords:
        stops = list('의가이은들는좀잘걍과도를자에와한것') + ['으로', '하다']
    else:
        stops = stopwords
    words = st.morphs(letters_only, stem=True, norm=True)
    meaningful_words = [w for w in words if w not in stops]
    return ' '.join(meaningful_words)


def processReviews(reviews, st, saveAs=None, saveOverride=False, 
                   do_preprocess=True, return_processed_review=False):
    import os
    import dill
    if not saveOverride and saveAs and os.path.isfile(saveAs):
        [wordOccurenceMatrix, self.vectorizer] = dill.load(open(saveAs, 'r'))
        return wordOccurenceMatrix
    if do_preprocess:
        processed_reviews = []
        for i, review in enumerate(reviews):
            if (i + 1) % 10000 == 0:
                print(' Review {} of {}'.format(i + 1, len(reviews)))
            processed_reviews.append(self.processSingleReview(review, st, i))
    else:
        processed_reviews = reviews
    if return_processed_review:
        return processed_reviews
    self.vectorizer = CountVectorizer(analyzer='word',
                                      tokenizer=None,
                                      preprocessor=None,
                                      max_features=MAX_VOCAB_SIZE)
    train_data_features = self.vectorizer.fit_transform(processed_reviews)
    wordOccurenceMatrix = train_data_features
    if saveAs:
        dill.dump([wordOccurenceMatrix, self.vectorizer], open(saveAs, 'w'))
    return wordOccurenceMatrix


class _SentimentLDABase:
    
    __config = dict(
        alpha=10,
        beta=0.1,
        gamma=0.1,
        n_topics=4,
        n_sentis=8
    )
    
    def __init__(self, config=None):
        if config is not None:
            if isinstance(config, dict):
                for key in self.config.keys():
                    if key not in config.keys():
                        raise AttributeError(f"config must have ``{key}``.")
            else:
                raise AttributeError(f"config must to be ``dict``.")
            self.config = config
        
    @property
    def config(self):
        return self.__config
    
    @config.setter
    def config(self, config):
        self.__config = config
    
    
class JST(_SentimentLDABase):
    
    def __init__(self, vocab, senti_dict=None, config=None):
        super().__init__(config)
        self.vocab = vocab
        if senti_dict is None:
            senti_dict = KSenticNet()
        self.senti_dict = senti_dict
    
    def _initialize_(self, WordOccurrenceMatrix, st):
        self.WordOccurrenceMatrix = WordOccurrenceMatrix
        n_docs, vocab_size = self.WordOccurrenceMatrix.shape
        
        # Pseudocounts
        self.n_dt = np.zeros((n_docs, self.n_topics))
        self.n_dts = np.zeros((n_docs, self.n_topics, self.n_sentis))
        self.n_d = np.zeros((n_docs))
        self.n_vts = np.zeros((vocab_size, self.n_topics, self.n_sentis))
        self.n_ts = np.zeros((self.n_topics, self.n_sentis))
        self.topics = {}
        self.sentiments = {}
        self.priorSentiment = {}
        
        alphaVec = self.alpha * np.ones(self.n_topics)
        gammaVec = self.gamma * np.ones(self.n_sentis)
        
        print('--* KSenticNet으로 사전 확률 조작 중... *--')s
        # 감정 사전 (KSenticNEt)을 사용하여 사전 확률을 조작 중.
        for i, word in enumerate(self.vectorizer.get_feature_names()):
            w = KSenticNet.keys.get(word)
            if not w: continue
            synsets = KSenticNet.scores[w, :]
            self.priorSentiment[i] = np.random.choice(self.numSentiments, p=synsets)
        
        print('--* initialize 작업 진행 중... *--')
        for d in range(numDocs):
            if d % 5000 == 0: print(' Doc {} of {} Reviews'.format(d, numDocs))
            topicDistribution = sample_from_dirichlet(alphaVec)
            sentimentDistribution = np.zeros((self.numTopics, self.numSentiments))
            for t in range(self.numTopics):
                sentimentDistribution[t, :] = sample_from_dirichlet(gammaVec)
            for i, w in enumerate(word_indices(self.wordOccurenceMatrix[d, :].toarray()[0])):
                t = sample_from_categorical(topicDistribution)
                s = sample_from_categorical(sentimentDistribution[t, :])
                
                self.topics[(d, i)] = t
                self.sentiments[(d, i)] = s
                self.n_dt[d, t] += 1
                self.n_dts[d, t, s] += 1
                self.n_d[d] += 1
                self.n_vts[w, t, s] += 1
                self.n_ts[t, s] += 1
                
    def conditionalDistribution(self, d, v):
        probabilites_ts = np.ones((self.numTopics, self.numSentiments))
        firstFactor = (self.n_dt[d] + self.alpha) / \
                (self.n_d[d] + self.numTopics * self.alpha)
        secondFactor = (self.n_dts[d, :, :] + self.gamma) / \
                (self.n_dt[d, :] + self.numSentiments * self.gamma)[:, np.newaxis]
        thirdFactor = (self.n_vts[v, :, :] + self.beta) / \
                (self.n_ts + self.n_vts.shape[0] * self.beta)
        probabilites_ts *= firstFactor[:, np.newaxis]
        probabilites_ts *= secondFactor * thirdFactor
        probabilites_ts /= np.sum(probabilites_ts)
        return probabilites_ts
                
    def run(self, WordOccurrenceMatrix, st, maxIters=30):
        self._initialize_(WordOccurrenceMatrix)
        numDocs, vocabSize = self.wordOccurenceMatrix.shape
        for iteration in range(maxIters):
            gc.collect()
            print('Starting iteration {} of {}'.format(iteration + 1, maxIters))
            for d in range(numDocs):
                for i, v in enumerate(word_indices(self.wordOccurenceMatrix[d, :].toarray()[0])):
                    t = self.topics[(d, i)]
                    s = self.sentiments[(d, i)]
                    self.n_dt[d, t] -= 1
                    self.n_d[d] -= 1
                    self.n_dts[d, t, s] -= 1
                    self.n_vts[v, t, s] -= 1
                    self.n_ts[t, s] -= 1
                    
                    probabilites_ts = self.conditionalDistribution(d, v)
                    if v in self.priorSentiment:
                        s = self.priorSentiment[v]
                        t = sample_from_categorical(probabilites_ts[:, s])
                    else:
                        ind = sample_from_categorical(probabilites_ts.flatten())
                        t, s = np.unravel_index(ind, probabilites_ts.shape)
                    
                    self.topics[(d, i)] = t
                    self.sentiments[(d, i)] = s
                    self.n_dt[d, t] += 1
                    self.n_d[d] += 1
                    self.n_dts[d, t, s] += 1
                    self.n_vts[v, t, s] += 1
                    self.n_ts[t, s] += 1
        print('Done.')