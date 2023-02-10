import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# data collection and data processing
# load dataset to dataframe

sonar_data = pd.read_csv('sonar_data.csv', header=None)
print(sonar_data.head())
print(sonar_data.shape)

print(sonar_data[60].value_counts())

# split into x and y (label) data set
# last col in dataset contains if object is a mine or a rock (M or R)
# drop coln -> axis = 1
# drop row -> axis = 0
x = sonar_data.drop(columns=60, axis=1)
y = sonar_data[60]

print(x)
print(y)

# split into training and test data
# stratify = y -> split data evently between labeled y data (R and M)
# random_state -> allows you to reproduce the code. If someone is using the same dataset and same random_state
# they will get the same results
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=1)

print(X_test.shape)
print(y_test.shape)


# train the model -> logistic regression (good to use to for binary classification)
model = LogisticRegression()

model.fit(X_train, y_train)

# evaluate the model
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)
print("Accuracy score on training data:", training_data_accuracy)


# accuracy on test data
x_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(x_test_prediction, y_test)
print("Accuracy score on test data:", testing_data_accuracy)

# make a predictive system
input_data = [X_train[5:6][:]]
print(input_data)
input_data_numpy = np.asarray(input_data)
print(input_data_numpy)

input_data_reshaped = input_data_numpy.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)

