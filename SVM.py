import pandas as pd 
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
print("Accuracy:", accuracy_score(y_test, y_pred))

end = time.perf_counter()
print("Time taken:", end-start)



