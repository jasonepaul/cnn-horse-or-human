# !/usr/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from tensorflow.keras import Sequential, layers, regularizers, models, optimizers
from sklearn.model_selection import train_test_split
import cv2
from sklearn.metrics import confusion_matrix, classification_report

# ********************************* Read in the data ********************************

# define folder paths for the data
train_humans_path = "../data/horse-or-human/train/humans/"
train_horses_path = "../data/horse-or-human/train/horses/"

paths = [train_humans_path, train_horses_path]

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
print(f"\nNumber of y_data labels that are ones (expected 527): {y_data.sum()}")

# ****************************** Scale features ****************************************

X_data = X_data / 255.0
print("\nThe first X_data sample after normalization: \n", X_data[0])

# ****************************** Split into train and test sets *************************

X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

print(f"\nShape of X_train: {X_train.shape}")
print(f"Shape of X_valid: {X_valid.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_valid: {y_valid.shape}\n")

# ****************************** Build model and fit ***********************************

# Build model
clf = Sequential()
clf.add(layers.Conv2D(16, (3, 3), strides=(1, 1), activation='relu', input_shape=(300, 300, 3)))
clf.add(layers.MaxPooling2D(2, 2))
clf.add(layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))
clf.add(layers.MaxPooling2D(2, 2))
clf.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
clf.add(layers.MaxPooling2D(2, 2))
clf.add(layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
clf.add(layers.MaxPooling2D(2, 2))
clf.add(layers.Flatten())
clf.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.03)))
# clf.add(layers.Dense(16, activation='relu'))
clf.add(layers.Dropout(0.5))
clf.add(layers.Dense(1, activation='sigmoid'))

# Print summary of model
clf.summary()

# Compile model and fit
opt = optimizers.Adam(lr=0.001)
clf.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
history = clf.fit(X_train, y_train, validation_split=0.2, epochs=25, verbose=2)

# ********************************** Save Model ***************************************

clf.save('model')

# ************************** Evaluate metrics on training data ************************

_, accuracy = clf.evaluate(X_train, y_train)
print('\nAccuracy on training data is: {:.1%}'.format(accuracy))

# ******************************* Visualize results ***********************************

params = {'legend.fontsize': 'x-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)

fig, axes = plt.subplots(2, figsize=(15, 15), sharex=True)
axes[0].plot(history.history['accuracy'])
axes[0].plot(history.history['val_accuracy'])
axes[0].set_title('Model Accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim(ymax=1.0)
axes[0].grid()
axes[0].legend(['Train', 'Validation'], loc='lower right')
axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_title('Model Loss')
axes[1].set_ylabel('Loss')
# axes[1].set_yscale("log")
axes[1].set_xlabel('Epoch')
axes[1].grid()
axes[1].legend(['Train', 'Validation'], loc='upper right')
plt.savefig('../results/fig_acc_loss.svg', dpi=400, bbox_inches='tight')

# ******************************* Load Model *************************************************

reconstructed_model = models.load_model('model')  # comment out if not loading saved model

# ******************************* Evaluate metrics on validation data *******************************************

_, accuracy = reconstructed_model.evaluate(X_valid, y_valid)  # only use when loading saved model
# _, accuracy = clf.evaluate(X_test, y_test)  # only use when using model in memory
print('\nAccuracy on validation hold-out set is: {:.1%}\n'.format(accuracy))

y_pred_valid = clf.predict_classes(X_valid)

# Confusion matrix and classification report
cm = confusion_matrix(y_valid, y_pred_valid)
print("Confusion matrix:\n", cm)

cr = classification_report(y_valid, y_pred_valid)
print("\nClassification report:\n", cr)
