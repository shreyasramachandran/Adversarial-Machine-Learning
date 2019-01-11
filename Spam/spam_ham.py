#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 10:26:59 2019

@author: shreyas
"""

import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


nltk.download('wordnet')

emails = pd.read_csv("emails.csv")
emails.head()

emails.at[58,'text']
emails.shape

# To see how many are spam and not spam
label_counts = emails.spam.value_counts()
plt.figure(figsize = (12,6))
sns.barplot(label_counts.index, label_counts.values, alpha = 0.9)

plt.xticks(rotation = 'vertical')
plt.xlabel('Spam', fontsize =12)
plt.ylabel('Counts', fontsize = 12)
plt.show()

# To check if length of email is corelated with spam and not spam
emails['length'] = emails['text'].map(lambda text: len(text))
emails.groupby('spam').length.describe()

# Checking distribution after setting a threshold
emails_subset = emails[emails.length < 1800]
emails_subset.hist(column='length', by='spam', bins=50)

# Tokenization
emails['tokens'] = emails['text'].map(lambda text:  nltk.tokenize.word_tokenize(text)) 

# Stop words removal
stop_words = set(nltk.corpus.stopwords.words('english'))
emails['filtered_text'] = emails['tokens'].map(lambda tokens: [w for w in tokens if not w in stop_words]) 

# Removing subject word
emails['filtered_text'] = emails['filtered_text'].map(lambda text: text[2:])


# Removing special characters form the email
emails['filtered_text'] = emails['filtered_text'].map(lambda text: ' '.join(text))
emails['filtered_text'] = emails['filtered_text'].map(lambda text: re.sub('[^A-Za-z0-9]+', ' ', text))

# Lemmatization
wnl = nltk.WordNetLemmatizer()
emails['filtered_text'] = emails['filtered_text'].map(lambda text: wnl.lemmatize(text))

# Wordcloud of spam mails
spam_words = ''.join(list(emails[emails['spam']==1]['filtered_text']))
spam_wordclod = WordCloud(width = 512,height = 512).generate(spam_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(spam_wordclod)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()

# Wordcloud of non-spam mails
spam_words = ''.join(list(emails[emails['spam']==0]['filtered_text']))
spam_wordclod = WordCloud(width = 512,height = 512).generate(spam_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(spam_wordclod)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(emails["filtered_text"],emails["spam"], test_size=0.2, random_state=10)

# Bag of words with naive bayes
count_vectorizer = CountVectorizer()
count_vectorizer.fit(train_X)
X_train_df = count_vectorizer.transform(train_X)
X_test_df = count_vectorizer.transform(test_X)

classifier = MultinomialNB(alpha=1.8)
classifier.fit(X_train_df, train_y)
pred = classifier.predict(X_test_df)
accuracy_score(test_y, pred)

# TF-IDF with naive bayes
 
tf = TfidfVectorizer()
tf.fit(train_X)
tfidf_train_X = tf.transform(train_X)
tfidf_test_X = tf.transform(test_X)

classifier = MultinomialNB(alpha = 0.04)
classifier.fit(tfidf_train_X, train_y)
pred = classifier.predict(tfidf_test_X)
accuracy_score(test_y, pred)


tok = nltk.tokenize.word_tokenize(string)
tok = [w for w in tok if not w in stop_words]
tok = tok[2:]
tok = ' '.join(tok)
tok = re.sub('[^A-Za-z0-9]+', ' ', tok)
tok = wnl.lemmatize(tok)
tok = [tok]
tok = tf.transform(tok)
classifier.predict(tok)

