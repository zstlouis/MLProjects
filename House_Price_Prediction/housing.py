import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# load the dataset
house_price_data = sklearn.datasets.fetch_california_housing()
print(house_price_data)

# convert data to a dataframe
housing_dataframe = pd.DataFrame(house_price_data.data, columns=house_price_data.feature_names)
print(housing_dataframe.head())

# add target price column to dataframe
housing_dataframe['price'] = house_price_data.target
print(housing_dataframe.head())

# check the number of rows and columns
print(housing_dataframe.shape)

# output key statistical values of the dataset
print(housing_dataframe.describe())

# check if any values are missing from dataset
print(housing_dataframe.isnull().sum())


# display heatmap to display positive and negative correlation between features
correlation = housing_dataframe.corr()
plt.figure(figsize=(8,8))

sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')

plt.show()

# split data variables from target variable
x = housing_dataframe.drop(columns='price', axis=1)
y = housing_dataframe['price']
print(x)
print(y)

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
print(X_train.shape)

# create XGBregressor model
model = XGBRegressor()

# train the model
model.fit(X_train, y_train)

# calculate accuracy and prediction of the model
train_prediction = model.predict(X_train)
print(train_prediction)

# calculate mean squared error
mse = metrics.mean_squared_error(train_prediction, y_train)
print("Mean squared error: ", mse)
