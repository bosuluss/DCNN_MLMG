# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 20:39:45 2021

@author: bosulus
"""

import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
import os
import seaborn as sns
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers.normalization import BatchNormalization
from keras import layers 
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras import Model
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.preprocessing.image import ImageDataGenerator


from keras.applications.vgg16 import VGG16, preprocess_input






#####################################################
# Read input images and assign labels based on folder names
print(os.listdir("EAC_Dataset/dataset20%/"))

SIZE = 224  #Resize images

#Capture training data and labels into respective lists
train_images = []
train_labels = [] 

for directory_path in glob.glob("EAC_Dataset/dataset20%/train/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)

#Convert lists to arrays        
train_images = np.array(train_images)
train_labels = np.array(train_labels)


# Capture test/validation data and labels into respective lists

test_images = []
test_labels = [] 
for directory_path in glob.glob("EAC_Dataset/dataset20%/validation/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(fruit_label)

#Convert lists to arrays                
test_images = np.array(test_images)
test_labels = np.array(test_labels)

#Encode labels from text to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

#Split data into test and train datasets (already split but assigning to meaningful convention)
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

###################################################################
# Scale pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

#One hot encode y values for neural network. Not needed for Random Forest
from keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

from tensorflow.keras.applications import vgg16
vgg_conv = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
vgg_conv.summary()

#Freeze all the layers
for layer in vgg_conv.layers:
    print(layer.name)
    layer.trainable = False 
#for layer in vgg_conv.layers:
    print(len(vgg_conv.layers))
    
    #we take the last layer of our the model and add it to our classifier

last = vgg_conv.get_layer('block5_pool')
print('last layer output shape:', last.output_shape)
last_output = last.output
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
#x = layers.GlobalAveragePooling2D()(last)
x = Flatten()(last)
# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)
# Add a dropout rate of 0.3
x = layers.Dropout(0.3)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(9, activation='softmax')(x)

# Configure and compile the model

model = Model(vgg_conv.input, x)
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
model.summary()

# We start the training

# We train it
history= model.fit(x_train, y_train_one_hot, epochs=100, validation_data = (x_test, y_test_one_hot))
# We evaluate the accuracy and the loss in the test set
scores = model.evaluate(X_test_resized, Y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

#Fine tuning 

base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
base_model.summary()

# We start the training

# We train the model
history= model.fit(x_train, y_train_one_hot, epochs=100, validation_data = (x_test, y_test_one_hot))

# We evaluate the accuracy and the loss in the test set
scores = model.evaluate(X_test_resized, Y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])



# We allow to the last convolutional andthe classification stages  to train
for layer in base_model.layers:
  if layer.name == 'block5_conv1':
    break
  layer.trainable = False
  print('Layer ' + layer.name + ' frozen.')
# We add our classificator (top_model) to the last layer of the model
last = base_model.layers[-1].output
x = Flatten()(last)
x = Dense(1000, activation='relu', name='fc1')(x)
x = Dropout(0.3)(x)
x = Dense(10, activation='softmax', name='predictions')(x)
model = Model(base_model.input, x)
# We compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# We see the new structure of the model
model.summary()




# Load the normalized images
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
# Change the batchsize according to your system RAM
train_batchsize = 100
val_batchsize = 10
# Data generator for training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=train_batchsize,
    class_mode='categorical')
# Data generator for validation data
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(image_size, image_size),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False)

# Configure the model for training

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=
         train_generator.samples/train_generator.batch_size,batch_size=()
         epochs=20,
         validation_data=validation_generator,
         validation_steps=
         validation_generator.samples/validation_generator.batch_size,
         verbose=1)
# Utility function for plotting of the model results
def visualize_results(history):
    # Plot the accuracy and loss curves
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
# Run the function to illustrate accuracy and loss
visualize_results(history)

    


# Utility function for obtaining of the errors

def obtain_errors(val_generator, predictions):
    # Get the filenames from the generator

    fnames = validation_generator.filenames
    # Get the ground truth from generator
    ground_truth = validation_generator.classes
    # Get the dictionary of classes
    label2index = validation_generator.class_indices
    # Obtain the list of the classes
    idx2label = list(label2index.keys())
    print("The list of classes: ", idx2label)
    # Get the class index
    predicted_classes = np.argmax(predictions, axis=1)
    errors = np.where(predicted_classes != ground_truth)[0]
    print("Number of errors = {}/{}".format(len(errors),validation_generator.samples))
    return idx2label, errors, fnames
# Utility function for visualization of the errors
def show_errors(idx2label, errors, predictions, fnames):
    # Show the errors
    for i in range(len(errors)):
        pred_class = np.argmax(predictions[errors[i]])
        pred_label = idx2label[pred_class]
        title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
            fnames[errors[i]].split('/')[0],
            pred_label,
            predictions[errors[i]][pred_class])
 
        original = load_img('{}/{}'.format(validation_dir,fnames[errors[i]]))
        plt.figure(figsize=[7,7])
        plt.axis('off')
        plt.title(title)
        plt.imshow(original)
        plt.show()

# Get the predictions from the model using the generator
predictions = model.predict(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
# Run the function to get the list of classes and errors
idx2label, errors, fnames = obtain_errors(validation_generator, predictions)
# Run the function to illustrate the error cases
show_errors(idx2label, errors, predictions, fnames)


i=0
for layer in base_model.layers:
    layer.trainable = False
    i = i+1
    print(i,layer.name)

x = base_model.output
x = Dense(128, activation='sigmoid')(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(args["9"], activation='softmax')(x)