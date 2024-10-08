import numpy as np 
import tensorflow as tf
from tensorflow.keras import layers 
import pandas as pd

train_data =pd.read_csv('training.csv')

X = train_data.drop('label',axis=1)
y = train_data['label']

X_train = train_data.drop('label',axis=1)
X_test = train_data.drop('label',axis=1)

y_train = train_data['label']
y_test = train_data['label']


