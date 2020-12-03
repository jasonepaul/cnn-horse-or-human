"""
Uses the VGG16 pre-trained model as a starting point. The fully connected
portion of the model is trained but not the filters.
"""
# !/usr/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# ********************************* Read in the data ********************************

# define folder paths for the data
train_humans_path = "../data/horse-or-human/train/humans/"
train_horses_path = "../data/horse-or-human/train/horses/"
validation_humans_path = "../data/horse-or-human/validation/humans/"
validation_horses_path = "../data/horse-or-human/validation/horses/"

paths = [train_humans_path, train_horses_path, validation_humans_path, validation_horses_path]
# subset path to only include the training data
paths = paths[:2]

X_data = []
y_data = []

for path in paths:
    for file in os.listdir(path):
        image = load_img(os.path.join(path, file), target_size=(224, 224))
        image = img_to_array(image)
        X_data.append(image)
        if 'human' in file:
            label = 1
        else:
            label = 0
        y_data.append(label)

X_data = np.asarray(X_data)
y_data = np.asarray(y_data)

# Next, the image pixels need to be prepared in the same way as the
# ImageNet training data was prepared. Specifically, from the paper
X_data = preprocess_input(X_data)

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

# Scaling is done using preprocess_input() as above
# X_data = X_data / 255.0
# print("\nThe first X_data sample after normalization: \n", X_data[0])

# ****************************** Split into train and test sets *************************

X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

print(f"\nShape of X_train: {X_train.shape}")
print(f"Shape of X_valid: {X_valid.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_valid: {y_valid.shape}\n")

# ****************************** Build model and fit ***********************************

# Build model
vgg_model = VGG16(include_top=False, input_shape=(224, 224, 3))
vgg_model.summary()

# freeze the convolutional layers so these don't train
for layer in vgg_model.layers:
    layer.trainable = False

# add new classifier layers (fully connected)
flat1 = Flatten()(vgg_model.layers[-1].output)
class1 = Dense(512, activation='relu')(flat1)
output = Dense(1, activation='sigmoid')(class1)
# define new model
vgg_model = Model(inputs=vgg_model.inputs, outputs=output)

# Print summary of model
vgg_model.summary()
# Compile model and fit
vgg_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = vgg_model.fit(X_train, y_train, validation_split=0.2, epochs=5, verbose=2)

# ********************************** Save Model ***************************************

vgg_model.save('vgg_model')

# ************************** Evaluate metrics on training data ************************

_, accuracy = vgg_model.evaluate(X_train, y_train)
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
plt.savefig('../results/fig_acc_loss_vgg.svg', dpi=400, bbox_inches='tight')

# ******************************* Load Model *************************************************

# reconstructed_model = models.load_model('vgg_model')  # comment out if not loading saved model

# ******************************* Evaluate metrics on validation data *******************************************

# _, accuracy = reconstructed_model.evaluate(X_valid, y_valid)  # only use when loading saved model
_, accuracy = vgg_model.evaluate(X_valid, y_valid)  # only use when using model in memory
print('\nAccuracy on validation hold-out set is: {:.1%}\n'.format(accuracy))
