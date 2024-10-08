import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

train_data =pd.read_csv('training.csv')

independent_variable = train_data.copy().drop("label", axis=1).values
dependent_variable = train_data["label"].values

xTrain, xTest, yTrain, yTest = train_test_split(independent_variable, dependent_variable, test_size=.2, random_state=153)
gnb = GaussianNB()
model = gnb.fit(xTrain, yTrain)
yPred = model.predict(xTest)
correct = 0
incorrect = 0
for j in range(0, yPred.size):
    if yPred[j] == yTest[j]:
        correct+=1
    else:
        incorrect+=1
    print("Prediction: ", yPred[j], " Real answer: ", yTest[j])
print("Correct: ", correct, " Incorrect: ", incorrect)