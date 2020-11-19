import pickle
import sys
from flask import (
    Flask, render_template, redirect, request, url_for
)
from bert import tokenizer, device
from bert import model as bert_model
from bert import predict as predict1
from jst import predict as predict2
from jst import embedding, jst_mb_model
app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/info')
def info():
    return render_template('info.html', title='Info')

@app.route('/about')
def about():
    return render_template('about.html', title='About')

@app.route('/analyze')
@app.route('/analyze/<sentence>')
def inputTest(sentence=''):
    if sentence is '':
        result1 = sentence
        result2 = sentence
    else:
        result1 = predict1(sentence, bert_model, device)
        result2 = predict2(sentence, embedding, jst_mb_model)
    return render_template(
        'analyze.html',
        sentence=sentence,
        result1=result1,
        result2=result2,
        title='Analyze'
    )

@app.route('/calculate', methods=['POST'])
def calculate():
    if request.method == 'POST':
        temp = request.form['sentence']
    else:
        temp = None
    print(f'cal: {temp}')
    return redirect(
        url_for(
            'inputTest',
            sentence=temp,
        )
    )

if __name__ == '__main__':
    app.run()
