import numpy as np 
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd

# importing training data
train_data =pd.read_csv('training.csv')

X = train_data.drop('label',axis=1)
y = train_data['label']

# Split dataset into training set and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_shape = [X_train.shape[1]]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu',
                         input_shape=input_shape),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mae')
losses = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=256,epochs=100)

print(losses)


