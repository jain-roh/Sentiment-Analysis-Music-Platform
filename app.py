
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textblob import TextBlob, Word
import pickle
import flask
from flask import render_template
import json
from flask import request, make_response

def lemmatize_with_postag(sentence):
    sent = TextBlob(sentence)
    tag_dict = {"J": 'a', 
                "N": 'n', 
                "V": 'v', 
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    return " ".join(lemmatized_list)
def remove_stopwords(review_words):
    all_texts=[]
    with open('vectorizer/stopwords_eng.txt') as stopfile:
        stopwords = stopfile.read()
        list = stopwords.split()
        for item in review_words.split():
            if not item in list:
                all_texts.append(item)
    return all_texts
# Lemmatize

file=open('vectorizer/tfidfvectorizer.pkl', 'rb')

loaded_vec = pickle.load(file)


# Use pickle to load in the pre-trained model.
app = flask.Flask(__name__, template_folder='templates')

# load weights into new model
@app.route('/', methods=['GET','POST'])
def loadmain():
    return render_template('index.html')
@app.route('/predict-sentiment', methods=['GET','POST'])
def main():
    if flask.request.method == 'GET':
        dataError = {}
        dataError['prediction']="Method Provided is GET. Please Try Using Post Method"
        json_data = json.dumps(dataError)
        resp = make_response(json_data)
        resp.status_code = 200
        resp.headers['Access-Control-Allow-Origin'] = '*'
        #prediction = 1
        return resp
    if flask.request.method == 'POST':
        text = flask.request.form['text']
        text=lemmatize_with_postag(text)
        all_texts=remove_stopwords(text)    

        model_path="model/"
        data = pd.Series(str(items) for items in [all_texts])


        x_input = loaded_vec.transform(data)
        loaded_model = mlflow.sklearn.load_model(model_path)
        yTPred=loaded_model.predict(x_input)
        rounded = [np.round(x) for x in yTPred]
        value={}
        if(rounded[0]==1):
            value['Prediction'] = 'Neutral'
        elif(rounded[0]==0):
            value['Prediction'] = 'Negative'
        else:
            value['Prediction'] = 'Positive'

        json_data = json.dumps(value)
        resp = make_response(json_data)
        resp.status_code = 200
        resp.headers['Access-Control-Allow-Origin'] = '*'
        #prediction = 1
        return resp
        #return render_template('main.html', result = data)
if __name__ == '__main__':
    app.run()
