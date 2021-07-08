# coding=utf-8
import re
import json
import numpy as np
from functools import wraps
from flask import (
    Flask, request, Response, jsonify
)
from bert import tokenizer, device
from bert import model as bert_model
from bert import predict as binary_predict
from jst import predict as multi_predict
from jst import embedding, jst_mb_model


app = Flask(__name__)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int):
            return int(obj)
        elif isinstance(obj, (np.float, np.float16, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def as_json(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        res = f(*args, **kwargs)
        res = json.dumps(res, ensure_ascii=False, cls=MyEncoder).encode('utf-8')
        return Response(res, content_type='application/json;charset=utf-8')
    return decorated_function


# @app.route('/predict', methods=['POST'])
# def predict():
#     return jsonify({'class_id': 'IMAGE_NET_XXX', 'class_name': 'Cat'})


def clean(s):
    s = s.replace("\n", "")
    s = s.replace("\t", "")
    s = re.sub(", +}", ",}", s)
    s = re.sub(", +]", ",]", s)
    s = s.replace(",}", "}")
    s = s.replace(",]", ",]")
    s = s.replace("'", "\"")
    return s


@app.route('/getSenti', methods=['POST'])
@as_json
def getSenti():
    data = request.get_data().decode('utf-8')
    data = clean(data)
    txts = json.loads(data)
    if 'texts' not in txts.keys():
        raise AttributeError("Key:'texts' is essential!")
    if not isinstance(txts['texts'], dict):
        raise AttributeError("Values must be dictionary")
    if not isinstance(list(txts['texts'].values())[0], str):
        raise AttributeError("")
    result1 = binary_predict(txts, bert_model, device)
    result2 = multi_predict(txts, embedding, jst_mb_model)
    results = dict(
        binary_results=result1,
        multi_results=result2
    )
    return results

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000", debug=False)
