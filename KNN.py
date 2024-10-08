import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# importing training data 
# only training up to 16800
train_data =pd.read_csv('training.csv')

independent_variable = train_data.copy().drop("label", axis=1).values
dependent_variable = train_data["label"].values

#print(independent_variable[0])
#print(dependent_variable[0])

xTrain, xTest, yTrain, yTest = train_test_split(independent_variable, dependent_variable, test_size=.2, random_state=153)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(xTrain, yTrain)
train_preds = knn_model.predict(xTest)
for j in range(0, train_preds.size):
    print("Prediction: ", train_preds[j], " Real answer: ", yTest[j])
