# SMILE
- **S**enti**M**ent **I**s **L**abeled **E**motions

임의의 data를 주어진 감정 token들로 감정 labeling 실시
- 인간의 8개 기본 감정으로
- t2snet으로 위 기능을 제공하고, 본 레포는 web service 제공용도로
    - https://github.com/jinmang2/t2snet
- 2021년도 12월까지 프로젝트 완료할 것 (석사 입학 전까지)

TO BE CONTINUED...

## 해결해야할 문제들
- chatspace dependency 제거
- bertopic 활용하여 임의의 데이터셋 labeling 기능 제공
- db는 sqlite 제공하기
- web ui 깔끔하게 만들기
- ai hub 감정 데이터 활용 방안 고려
- 대화 도메인 주제로 프로젝트 진행 (김수환님 발표 참고)
- 분류 모델을 naive bayes, bert에서 현재까지 내가 공부한 huggingface plm으로 교체 (배포된 모델로 분류 가능하게)
    ```python
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_name = "" # 앞으로 개발할 것.
    tokenizer = AutoTokenizer.from_pretrained(f"jinmang2/{model_name}")
    pt_model = AutoModelForSequenceClassification(f"jinmang2/{model_name}")
    ```
- DL기반 Topic Modeling으로 어떤 데이터가 어떻게 감정 군집화가 되었는지 분석 방안 제공
    - BERTopic으로 PLM 활용
    - W-LDA로도 분석 실시

## reference
### postman
- https://m.blog.naver.com/myohyun/222050987747
- https://stackoverflow.com/questions/49389535/problems-with-flask-and-bad-request
- https://tutorials.pytorch.kr/intermediate/flask_rest_api_tutorial.html
- https://toughrogrammer.tistory.com/222
- https://apt-info.github.io/%EA%B0%9C%EB%B0%9C/python-flask3-post/
- http://b1ix.net/402
