import numpy
import os
from tensorflow import keras
# from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Activation, Dropout, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras import backend as k
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

img_height, img_width = 224, 224
training_data_dir = 'train_data\\train'
testing_data_dir = 'train_data\\test'
batch_size = 100
epochs = 4
num_classes = 2

training_files = 0
testing_files = 0

for sub_folder in os.listdir(training_data_dir):
    path, dirs, files = next(os.walk(os.path.join(training_data_dir,sub_folder)))
    training_files += len(files)

for sub_folder in os.listdir(testing_data_dir):
    path, dirs, files = next(os.walk(os.path.join(testing_data_dir,sub_folder)))
    testing_files += len(files)

training_files, testing_files
#Instantiation
model = Sequential()
#1st convolutional layer
model.add(Conv2D(filters=96, input_shape=(img_height,img_width,3), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))        # Max pooling

#2nd convolutional layer
model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))        # Max pooling

#3rd convolutional layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

#4th convolutional layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

#5th convolutional layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))        # Max pooling

#Passing to fully connected layer
model.add(Flatten())

#1st fully connected layer
model.add(Dense(4096, input_shape=(img_height,img_width,3,)))
model.add(Activation('relu'))

model.add(Dropout(0.4))        # Add Dropout to prevent overfitting

#2nd fully connected layer
model.add(Dense(4096))
model.add(Activation('relu'))

model.add(Dropout(0.4))        # Add Dropout to prevent overfitting

#3rd fully connected layer
model.add(Dense(1000))
model.add(Activation('relu'))

#Output layer
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

model.compile(loss = keras.losses.categorical_crossentropy,
                    optimizer = 'adam',
                    metrics=["accuracy"])

# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.1,
width_shift_range = 0.1,
height_shift_range=0.1,
rotation_range=5)

test_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.1,
width_shift_range = 0.1,
height_shift_range=0.1,
rotation_range=5)

train_generator = train_datagen.flow_from_directory(
training_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size,
class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
testing_data_dir,
target_size = (img_height, img_width),
class_mode = "categorical")

# Save the model according to the conditions
checkpoint = ModelCheckpoint("car1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, verbose=1, mode='min')
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=0.0001)

### Start training!

Model.fit(model, x=train_generator,
           epochs=epochs,
           callbacks=[checkpoint, early, rlr],
           steps_per_epoch=training_files//batch_size,
           validation_data=validation_generator,
           validation_steps=testing_files//batch_size)

import matplotlib.pyplot as plt
print(model.history.history.keys())
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.ylim((0.8,1.0))
plt.grid()
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.ylim((0,0.5))
plt.grid()
plt.legend(['train', 'test'], loc='upper left')
plt.show()

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from keras.models import load_model
model = load_model("car1.h5")

# Make predictions on the test data
predictions = model.predict(validation_generator)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = validation_generator.classes

# Compute precision, recall, and confusion matrix
print("Confusion Matrix:")
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print(conf_matrix)

# Precision and Recall
class_labels = list(validation_generator.class_indices.keys())
print("\nClassification Report:")
class_report = classification_report(true_labels, predicted_labels, target_names=class_labels)
print(class_report)
