# !/usr/bin/python
import os
import numpy as np
import tensorflow as tf
import cv2
from sklearn.metrics import confusion_matrix, classification_report

# ********************************* Read in the data ********************************

# define folder paths for the data
validation_humans_path = "../data/horse-or-human/validation/humans/"
validation_horses_path = "../data/horse-or-human/validation/horses/"

paths = [validation_humans_path, validation_horses_path]

X_data = []
y_data = []

for path in paths:
    for file in os.listdir(path):
        image = cv2.imread(os.path.join(path, file), cv2.IMREAD_COLOR)
        X_data.append(image)
        if 'human' in file:
            label = 1
        else:
            label = 0
        y_data.append(label)

X_data = np.asarray(X_data)
y_data = np.asarray(y_data)

# *************************************** Inspect features and labels *****************

print("\nShape of X_data:", X_data.shape)
print("Shape of y_data:", y_data.shape)

print("\nData type of X_data:", type(X_data))
print("Data type of y_data:", type(y_data))

print("\nData type of 1st sample in X_data:", type(X_data[0]))
print("Data type of 1st label in y_data:", type(y_data[0]))

print("\nThe first X_data sample: \n", X_data[0])

print("\nThe y_data labels: \n", y_data)
print(f"\nNumber of y_data labels that are ones (expected 128): {y_data.sum()}")

# ****************************** Scale features ****************************************

X_data = X_data / 255.0
print("\nThe first X_data sample after normalization: \n", X_data[0])

# ****************************** Assign test sets *************************

X_test, y_test = X_data, y_data

print(f"\nShape of X_test: {X_test.shape}")
print(f"Shape of y_test: {y_test.shape}")

# ******************************* Load Model *************************************************

reconstructed_model = tf.keras.models.load_model('model')  # comment out if not loading saved model

# ******************************* Evaluate metrics on validation data *******************************************

_, accuracy = reconstructed_model.evaluate(X_test, y_test)  # only use when loading saved model
print('\nAccuracy on test set is: {:.1%}\n'.format(accuracy))

y_pred_test = reconstructed_model.predict_classes(X_test)

# Confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred_test)
print("Confusion matrix:\n", cm)

cr = classification_report(y_test, y_pred_test)
print("\nClassification report:\n", cr)
