import pandas as pd
import numpy as np 
import time 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

# Model Accuracy 
print("Test Set Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

# Full training set for final predictions 
rf.fit(X, y)

# Load test data 
test_data =pd.read_csv('testing.csv')

# Predictions for test data 
test_predictions = rf.predict(test_data)

# Save predictions 
with open('RF_predictions.txt', 'w') as f:
    f.write("Test Label\n")
    for prediction in test_predictions:
        f.write(f"{prediction}\n")

end = time.perf_counter()
print("Time taken:", end-start)

