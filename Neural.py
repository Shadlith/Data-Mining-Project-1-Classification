import numpy as np 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.utils import to_categorical
import pandas as pd
import time 

start=time.perf_counter()

# importing training data
train_data =pd.read_csv('training.csv')

X = train_data.drop('label',axis=1)
y = train_data['label']

# Split dataset into training set and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0

num_classes = 10
y_train_encoded = to_categorical(y_train, num_classes)
y_test_encoded = to_categorical(y_test, num_classes)

neurons_per_layer = 512
dropout = 0.3

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=neurons_per_layer, activation='relu',
                         input_shape=[X_train.shape[1]]),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(units=neurons_per_layer, activation='relu'),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(units=neurons_per_layer, activation='relu'),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(units=neurons_per_layer, activation='relu'),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])


# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)

# Train Neural Network Classifier
losses = model.fit(X_train, y_train_encoded, validation_data=(X_test, y_test_encoded), batch_size=128, epochs=200, callbacks=[early_stopping_cb])
print(losses.history)

# Predict the response for test dataset
y_pred_encoded = model.predict(X_test)
y_pred = np.argmax(y_pred_encoded, axis=1)

# Model Accuracy 
print("Neural Network Test Set Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

# Full training set for final predictions 
X_full = X / 255.0
y_full_encoded = to_categorical(y, num_classes)
#model.fit(X_full, y_full_encoded, batch_size=256, epochs=100, verbose=0)

# Load test data 
test_data = pd.read_csv('testing.csv')
X_test_final = test_data / 255.0

# Test data predictions 
test_predictions_encoded = model.predict(X_test_final)
test_predictions = np.argmax(test_predictions_encoded, axis=1)

# Save predictions
with open('Neural_predictions.txt', 'w') as f:
    f.write('Test Label\n')
    for prediction in test_predictions:
        f.write(f"{prediction}\n")

end = time.perf_counter()
print("Time taken:", end-start)

print()