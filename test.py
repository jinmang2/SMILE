from bert import tokenizer, device
from bert import model as bert_model
from bert import predict as binary_predict
from jst import predict as multi_predict
from jst import embedding, jst_mb_model

sentence = {
    'texts': {
        '200': '폴킴의 노래는 언제나 날 행복하게 해준다',
        '10': '감정분석은 완벽해!',
        '50': '열받게하네?'
    }
}
result1 = binary_predict(sentence, bert_model, device)
result2 = multi_predict(sentence, embedding, jst_mb_model)

results = dict(
    binary_results=result1,
    multi_results=result2
)

print(sentence)
print(results)
