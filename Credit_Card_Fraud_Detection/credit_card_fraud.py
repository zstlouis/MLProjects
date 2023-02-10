import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# dataset is very unbalanced. Will need to balance for accurate prediction

# load the dataset into a dataframe
credit_card_dataset = pd.read_csv('creditcard.csv')
print(credit_card_dataset.head())


# check if any data is missing
print(credit_card_dataset.isnull().sum())

# check count of fraudulent and legit transactions
# 0 -> legit transactions
# 1 -> Fraudulent transaction
print(credit_card_dataset['Class'].value_counts())
legit = credit_card_dataset[credit_card_dataset['Class'] == 0]
fraud = credit_card_dataset[credit_card_dataset['Class'] == 1]
print(legit.shape)
print(fraud.shape)


# build a sample dataset to have similar distribution between
# both legit and fraudulent transactions
# retrieve 492 legit transactions to match the same amount of fraudulent ones
# will then need to combine both the new legit and fradulent datasets together to run analysis on
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)
print(new_dataset.head())

# separate feature variable from target variable for classification
x = new_dataset.drop(columns='Class', axis=1)
y = new_dataset['Class']
print(x.head())
print(y.head())


# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)


# standardize the data
scaler = StandardScaler()
train_std = scaler.fit_transform(X_train)
test_std = scaler.transform(X_test)
# create logistic regression model
model = LogisticRegression()
model.fit(train_std, y_train)

# retrieve accuracy of the model
train_pred = model.predict(train_std)
train_acc = accuracy_score(train_pred, y_train)
print("Accuracy of the model against the training dataset: ", train_acc)

test_pred = model.predict(test_std)
test_acc = accuracy_score(test_pred, y_test)
print("Accuracy of the model against the testing dataset: ", test_acc)
