from functools import partial
from flask import Flask, url_for, request
from nltk.corpus import wordnet
app = Flask(__name__)


@app.route('/synset', methods=['POST', 'GET'])
def getSynset():

    def make_synset_generator(func):
        def wrapper_function(synsets, *args, **kwargs):
            if isinstance(synsets, list):
                for synset in synsets:
                    yield from func(synset, *args, **kwargs)
            else:
                yield from func(synsets, *args, **kwargs)
        return wrapper_function

    def get_name(s):
        s = s.name().split('.')[0]
        s = s.replace('_', ' ')
        return s

    @make_synset_generator
    def hypernyms_generator(synset):
        yield from synset.hypernyms()

    @make_synset_generator
    def hyponyms_generator(synset):
        yield from synset.hyponyms()

    # 정답과 사용자 인풋 텍스트 받아오기
    correct_ = request.args.get('correct')
    input_ = request.args.get('input')

    if type(input_) == type(None):
        return "User input is 'None'"
    if type(correct_) == type(None):
        return "Correct answer is 'None'"

    zero_synsets = wordnet.synsets(correct_)
    if zero_synsets == []:
        return 'Fail'

    for nym in hypernyms_generator(zero_synsets):
        if get_name(nym) == input_:
            return 'Correct'

    for nym in hyponyms_generator(zero_synsets):
        if get_name(nym) == input_:
            return 'Correct'
        for nym2 in hyponyms_generator(nym):
            if get_name(nym2) == input_:
                return 'Correct'
            for nym3 in hyponyms_generator(nym2):
                if get_name(nym3) == input_:
                    return 'Correct'
    return 'Fail'


if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8080")
