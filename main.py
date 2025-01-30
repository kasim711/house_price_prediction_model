'''
1. House Price Prediction

-> Goal: Predict house prices based on features like size, location, and number of rooms.
-> Tech: Linear Regression, Python (libraries: Pandas, NumPy, scikit-learn).
-> Data: Kaggle datasets or open-source housing data.
'''
# installing numpy, pandas, matplotlib, seaborn, scikit-learn in command client
# california housing prices - cam nugent - kaggle dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split # comes in use after line 20
from sklearn.linear_model import LinearRegression # after line 54
from sklearn.preprocessing import StandardScaler # after line 73(mainly 57)
from sklearn.ensemble import RandomForestRegressor # after line 78
from sklearn.model_selection import GridSearchCV # after line 84
data = pd.read_csv("housing.csv")
# print(data.info())
data.dropna(inplace=True)

x = data.drop(['median_house_value'], axis=1)
y = data['median_house_value']
# print(x)
# print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
train_data = x_train.join(y_train)
# print(train_data)
train_data.hist(figsize=(15,8))
# plt.show() # for showing diagrams(graphs)
plt.figure(figsize=(15,8))
# sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")
# plt.show()
train_data["total_rooms"] = np.log(train_data["total_rooms"] + 1)
train_data["total_bedrooms"] = np.log(train_data["total_bedrooms"] + 1)
train_data["population"] = np.log(train_data["population"] + 1)
train_data["households"] = np.log(train_data["households"] + 1)
train_data.hist(figsize=(15,8))
# plt.show()
# print(train_data.ocean_proximity.value_counts())
train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(["ocean_proximity"], axis=1)
# print(train_data)

plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")
# plt.show() 
plt.figure(figsize=(15,8))
sns.scatterplot(x="latitude", y="longitude", data=train_data,hue="median_house_value",palette="coolwarm")
# plt.show() 
train_data["bedroom_ratio"] = train_data["total_bedrooms"] / train_data["total_rooms"]
train_data["household_rooms"] = train_data["total_rooms"] / train_data["households"]
plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")
# plt.show()

scaler = StandardScaler()
x_train, y_train = train_data.drop(["median_house_value"], axis=1), train_data["median_house_value"]
x_train_s = scaler.fit_transform(x_train)

reg = LinearRegression() 
reg.fit(x_train_s, y_train)

test_data = x_test.join(y_test)

test_data["total_rooms"] = np.log(test_data["total_rooms"] + 1)
test_data["total_bedrooms"] = np.log(test_data["total_bedrooms"] + 1)
test_data["population"] = np.log(test_data["population"] + 1)
test_data["households"] = np.log(test_data["households"] + 1)

test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(["ocean_proximity"], axis=1)

test_data["bedroom_ratio"] = test_data["total_bedrooms"] / test_data["total_rooms"]
test_data["household_rooms"] = test_data["total_rooms"] / test_data["households"]

x_test, y_test = test_data.drop(["median_house_value"], axis=1), test_data["median_house_value"]
x_test_s = scaler.transform(x_test)
# print(reg.score(x_test_s, y_test))

forest = RandomForestRegressor()

forest.fit(x_train_s, y_train)
# print(forest.score(x_test_s, y_test))


forest = RandomForestRegressor()
param_grid = {
    "n_estimators": [100,200,300],
    "min_samples_split": [2,4],
    "max_depth": [None,4,8] 
}

grid_search = GridSearchCV(forest, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)
grid_search.fit(x_train_s, y_train)
# print(grid_search.fit(x_train_s, y_train))

# print(grid_search.best_estimator_)

print(grid_search.best_estimator_.score(x_test_s, y_test))
