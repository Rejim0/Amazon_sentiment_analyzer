import os

from flask import Flask,render_template,redirect,request,url_for
import pickle
from textblob import TextBlob
from utils import text_preprocess


app = Flask(__name__)


model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html',result = None)

@app.route('/predict',methods=['POST'])
def predict():
    text = request.form.get('review')
    polarity_text = TextBlob(text).sentiment.polarity
    subjectivity_text = TextBlob(text).sentiment.subjectivity
    preprocess_text = text_preprocess(text)
    vectorized_text = vectorizer.transform([preprocess_text])
    result_predict = model.predict(vectorized_text)
    if result_predict[0] == 0:
        result = "Negative"
    else:
        result = "Positive"
    return render_template('index.html',
                           result=result,
                           polarity=round(polarity_text,2),
                           subjectivity=round(subjectivity_text,2),
                           review_text=text)


if __name__ == '__main__':
    port = int(os.environ.get('PORT',5000))
    app.run(host = '0.0.0.0',port = port, debug=True)
