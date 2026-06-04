# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 22:04:00 2026

@author: Admin
"""

# We are importing tools that help us do ML
# Think of these like importing a calculator before doing math

import pandas as pd                          # for handling data (like Excel in Python)
import re                                    # for cleaning text
import nltk                                  # Natural Language Toolkit
nltk.download('stopwords')                   # download list of common words like "the", "is"
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer   # converts text to numbers
from sklearn.linear_model import LogisticRegression           # ML model
from sklearn.svm import LinearSVC                             # another ML model
from sklearn.metrics import accuracy_score, classification_report

#load the dataset
def load_data(filepath):
      data = [] #empty list to store rows
      with open('train_data.txt', 'r' , encoding='utf-8') as f:
          for line in f : # read line by line
              parts = line.strip().split(' ::: ') # splitting
              if len(parts) == 4:
                  data.append({
                      'id': parts[0],
                      'title': parts[1],
                      'genre' : parts[2],
                      'plot' : parts[3]
                      })
      return pd.DataFrame(data)
              
#load train and test file
train_df=load_data('train_data.txt')
test_df= load_data('test_data.txt')

print(train_df.shape)
print(train_df.head())
print(train_df['genre'].value_counts()) # how many movies per genre


#text cleaning

stop_words=set(stopwords.words('english'))
# stop_wprds = the ,is , a ,an, in

def clean_text(text):
    text=text.lower()
    text = re.sub(r'[^a-z\s]', ' ' , text) #remove punctuation &numbers
    words = text.split()
    words = [w for w in words if w not in stop_words] #remove stop words
    return ' '.join(words)

#after cleaning to every plot summary
train_df['clean_plot']= train_df['plot'].apply(clean_text)
test_df['clean_plot']  = test_df['plot'].apply(clean_text)

#see difference
print("before", train_df['plot'][0])
print("after",train_df['plot'][0])


#convert text to numbers

tfidf = TfidfVectorizer (max_features=15000, ngram_range=(1,2), sublinear_tf =True)
 #max_features=15000 -> only keep top 15000 words
 #ngram_range=(1,2) -> consider single words AND  pairs
 
X_train = tfidf.fit_transform(train_df['clean_plot'])
X_test = tfidf.transform(test_df['clean_plot'])

y_train = train_df['genre'].str.strip()
print("shape of X_train:" , X_train.shape)

#training the model

model = LinearSVC(C=1.0, max_iter=2000)
# LinearSVC = Support Vector Machine — good for text classification
# C = how strictly it tries to fit training data (1.0 is a safe default)

model.fit(X_train, y_train) # here learning happens

print("model trained successfully")

# evaluating the model

from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=42
    )

model.fit(X_tr, y_tr)
predictions = model.predict(X_val)

print("Accuracy", accuracy_score(y_val, predictions))
print(classification_report(y_val, predictions , zero_division=0))

model.fit(X_train, y_train)
test_preds = model.predict(X_test)

test_df['predicted genre'] = test_preds
print(test_df[['title', 'predicted genre']].head(10))

test_df[['id', 'title', 'predicted genre']].to_csv('predictions.csv', index=False)
print("✅ Saved to predictions.csv")