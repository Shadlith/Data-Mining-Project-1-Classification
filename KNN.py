import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

import time 

start=time.perf_counter()

# importing training data 
# only training up to 16800
train_data =pd.read_csv('training.csv')

independent_variable = train_data.copy().drop("label", axis=1).values
dependent_variable = train_data["label"].values

#print(independent_variable[0])
#print(dependent_variable[0])

# Split dataset into training set and test set 
xTrain, xTest, yTrain, yTest = train_test_split(independent_variable, dependent_variable, test_size=.2, random_state=153)

# create knn classifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(xTrain, yTrain)

# Predict the response for test dataset
train_preds = knn_model.predict(xTest)

# Model Accuracy 
print("Test Set Metrics:")
print("Accuracy:", accuracy_score(yTest, train_preds))
print("Precision:", precision_score(yTest, train_preds, average='weighted'))
print("Recall:", recall_score(yTest, train_preds, average='weighted'))
print("F1 Score:", f1_score(yTest, train_preds, average='weighted'))

# Full training set for final predictions 
knn_model.fit(independent_variable, dependent_variable)

# Load test data 
test_data =pd.read_csv('testing.csv')

# Predictions for test data 
test_predictions = knn_model.predict(test_data)

# Save predictions 
with open('KNN_predictions.txt', 'w') as f:
    f.write("Test Label\n")
    for prediction in test_predictions:
        f.write(f"{prediction}\n")

end = time.perf_counter()
print("Time taken:", end-start)

# Print predictions
#for j in range(0, train_preds.size):
#    print("Prediction: ", train_preds[j], " Real answer: ", yTest[j])
