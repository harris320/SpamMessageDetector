import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, request, jsonify, render_template
import pickle
import re

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('cv-transform.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    message = request.form['message']
    
    data = [message]
        
    X = cv.transform(data).toarray()
        
    pred = model.predict(X)
    
    if pred==0:
        output = "Not spam"
    else:
        output = "SPAM"
    
    
    return render_template('index.html', prediction_text='The message is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)