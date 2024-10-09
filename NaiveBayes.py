import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import time 

start=time.perf_counter()
train_data =pd.read_csv('training.csv')

independent_variable = train_data.copy().drop("label", axis=1).values
dependent_variable = train_data["label"].values

xTrain, xTest, yTrain, yTest = train_test_split(independent_variable, dependent_variable, test_size=.2, random_state=153)
gnb = GaussianNB()
model = gnb.fit(xTrain, yTrain)
yPred = model.predict(xTest)

# Model Accuracy 
print("Test Set Metrics:")
print("Accuracy:", accuracy_score(yTest, yPred))
print("Precision:", precision_score(yTest, yPred, average='weighted'))
print("Recall:", recall_score(yTest, yPred, average='weighted'))
print("F1 Score:", f1_score(yTest, yPred, average='weighted'))

# Full training set for final predictions 
gnb.fit(independent_variable, dependent_variable)

# Load test data 
test_data =pd.read_csv('testing.csv')

# Predictions for test data 
test_predictions = gnb.predict(test_data)

# Save predictions 
with open('GNB_predictions.txt', 'w') as f:
    f.write("Test Label\n")
    for prediction in test_predictions:
        f.write(f"{prediction}\n")


end = time.perf_counter()
print("Time taken:", end-start)

""" 
correct = 0
incorrect = 0
for j in range(0, yPred.size):
    if yPred[j] == yTest[j]:
        correct+=1
    else:
        incorrect+=1
    print("Prediction: ", yPred[j], " Real answer: ", yTest[j])
print("Correct: ", correct, " Incorrect: ", incorrect) """