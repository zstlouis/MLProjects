import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# store csv data into a dataframe
diabetes_dataframe = pd.read_csv('diabetes.csv')
print(diabetes_dataframe.head())
print(diabetes_dataframe.describe())

# check if any data is missing
print(diabetes_dataframe.isna().sum())

# view the count of each outcome result
# 0 -> non diabetic
# 1 -> diabetic

print(diabetes_dataframe['Outcome'].value_counts())

# calculate avgs for each possible outcome value
print(diabetes_dataframe.groupby('Outcome').mean())

# separate labels from variable data
x = diabetes_dataframe.drop(columns='Outcome', axis=1)
print(x)
y = diabetes_dataframe['Outcome']
print(y)

# standardize the date
scaler = StandardScaler()
scaler.fit(x)
standardize_data = scaler.transform(x)

print(standardize_data)

X_train, X_test, y_train, y_test = train_test_split(standardize_data, y, test_size=0.2, stratify=y, random_state=2)
print(X_train.shape)

# create the model
# will be using support vector machine
# SVC -> support vector classifier
model = svm.SVC(kernel='linear')
model.fit(X_train,y_train)

# calculate the accuracy against the training data
# X_train_prediction = model.predict(X_train)
# training_data_accuracy = accuracy_score(X_train_prediction, y_train)

train_predict = model.predict(X_train)
train_acc = accuracy_score(train_predict, y_train)
print("Accuracy of model on training dataset: ", train_acc)

# calculate accuracy against the test data
test_predict = model.predict(X_test)
test_acc = accuracy_score(test_predict, y_test)
print("Accuracy of model on testing dataset: ", test_acc)

# Prediction output for a given row
data = [X_train[1:2][:]]
data_numpy = np.asarray(data)
data_reshape = data_numpy.reshape(1, -1)

# standardize the input data
std_data = scaler.transform(data_reshape)
pred = model.predict(std_data)

if pred[0] == 0:
    print("No diabetes")
else:
    print("Diabetes")