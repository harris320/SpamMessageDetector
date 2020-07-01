# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 12:11:59 2020

@author: Harris
"""

import pandas as pd

messages = pd.read_csv('SMSSpamCollection' , sep='\t', names=["label", "message"])

import re
import nltk

import pickle

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet = WordNetLemmatizer()

corpus = []

for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ' , messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()

y =  pd.get_dummies(messages['label'])
y = y.iloc[:, 1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

# y_pred = spam_detect_model.predict(X_test)

# from sklearn.metrics import  classification_report, confusion_matrix

# class_report = classification_report(y_test, y_pred)
# conf_mat= confusion_matrix(y_test, y_pred)

pickle.dump(cv, open('cv-transform.pkl', 'wb'))

pickle.dump(spam_detect_model, open('model.pkl','wb'))

#model = pickle.load(open('model.pkl','rb'))

#new_pred = model.predict(X_test)
