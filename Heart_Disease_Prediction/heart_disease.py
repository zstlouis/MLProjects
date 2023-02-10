import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('heart_disease_data.csv')
print(df.head())

print(df.describe())
print(df['chol'].mean())

# split into variables and labels
x = df.drop(columns='target', axis=1)
y = df['target']

print(x.shape)
print(y.shape)

# split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

print(X_train.shape, y_test.shape)

# create the model
model = LogisticRegression()

# train the model
model.fit(X_train, y_train)

# calc accuracy of the model on training data
train_predict = model.predict(X_train)
train_acc = accuracy_score(train_predict, y_train)
print("Accuracy of model on training dataset: ", train_acc)

# calc accuracy of model on test data
test_predict = model.predict(X_test)
test_acc = accuracy_score(test_predict, y_test)
print("Accuracy of the model on the test dataset: ", test_acc)

# print out a predicted value
# 1 => heart disease
# 0 => no heart disease

data = [X_train[206:207][:]]
data_numpy = np.asarray(data)
data_reshape = data_numpy.reshape(1, -1)
pred = model.predict(data_reshape)

if pred[0] == 0:
    print("No heart disease")
else:
    print("heart disease")

