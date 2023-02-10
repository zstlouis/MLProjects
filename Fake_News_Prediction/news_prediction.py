import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

# get list of english stop words to be removed from data
nltk.download('stopwords')

# print(stopwords.words('english'))

# load data into dataframe

news_dataset = pd.read_csv('fake-news/train.csv')
print(news_dataset.shape)
print(news_dataset.head())

# check if any data is missing
print(news_dataset.isnull().sum())

news_dataset = news_dataset.fillna('')
print(news_dataset.isnull().sum())

# main fields to analyze are author and title
# merge the fields under a new column named content
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']
print(news_dataset['content'])

# separate variable columns (x) from label column (y)
# x = news_dataset.drop(columns=['label'], axis=1)
# y = news_dataset['label']
# print(x)
# print(y)

# Stemming is the process of reducing a word to its root word
# example: actor, actress, acting -> act
port_stem = PorterStemmer()


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# apply to dataset column content
news_dataset['content'] = news_dataset['content'].apply(stemming)

# print(news_dataset.head())

# separate variable columns (x) from label column (y)
x = news_dataset['content'].values
y = news_dataset['label'].values
print(x)
print(y)

# convert text to numerical data
# vectorize
vectorizer = TfidfVectorizer()
vectorizer.fit(x)
x = vectorizer.transform(x)
print(x)

# split into training and testing
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# create and fit the model
model = LogisticRegression()
model.fit(X_train, y_train)

# calculate the accuracy score of the model against the training set
x_train_prediction = model.predict(X_train)
train_accuracy = accuracy_score(x_train_prediction, y_train)
print("Accuracy against the training dataset: ", train_accuracy)

# calculate the accuract score of the model against the testing set
x_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(x_test_prediction, y_test)
print("Accuracy against the testing dataset: ", test_accuracy)


# have model make a prediction for a single article
article = X_test[3]
prediction = model.predict(article)
if prediction[0] == 0:
    print('The news in this article is real.')
else:
    print('The news in this article is fake.')

