import pandas as pd 
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time 

start=time.perf_counter()

# importing training data
train_data =pd.read_csv('training.csv')

X = train_data.drop('label',axis=1)
y = train_data['label']

# Split dataset into training set and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a svm Classifier
clf = svm.SVC(kernel='linear')

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy 
print("Test Set Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

# Full training set for final predictions 
clf.fit(X, y)

# Load test data 
test_data =pd.read_csv('testing.csv')

# Predictions for test data 
test_predictions = clf.predict(test_data)

# Save predictions 
with open('SVM_predictions.txt', 'w') as f:
    f.write("Test Label\n")
    for prediction in test_predictions:
        f.write(f"{prediction}\n")

end = time.perf_counter()
print("Time taken:", end-start)


