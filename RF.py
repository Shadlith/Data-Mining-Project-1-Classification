import pandas as pd
import numpy as np 
import time 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



start=time.perf_counter()

# importing training data
train_data =pd.read_csv('training.csv')

X = train_data.drop('label',axis=1)
y = train_data['label']

# Split dataset into training set and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create random forest classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

end = time.perf_counter()
print("Time taken:", end-start)

