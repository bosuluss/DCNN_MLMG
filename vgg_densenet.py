# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 12:59:51 2020

@author: bosulus
This code explains the process of using pretrained weights (VGG16) 
as feature extractors for traditional machine learning classifiers (Random Forest).
"""

import os
import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import seaborn as sns
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D 
from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16 as VGG
from keras.applications.densenet import DenseNet201 as DenseNet
from keras.optimizers import SGD
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
#from image_functions import preprocessing_image_rgb 


# Read input images and assign labels based on folder names
print(os.listdir("EAC_Dataset/dataset20%"))

SIZE = 64  #Resize images

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
# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

#One hot encode y values for neural network. 
from keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

#############################
use_vgg= True
#Load model wothout classifier/fully connected layers
if use_vgg:
    VGG_model = VGG(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
else:
    VGG_model = DenseNet(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

top_model= VGG_model.output
top_model= GlobalAveragePooling2D()(top_model)
#top_model = Flatten()(top_model)

if use_vgg:
    #top_model= Dense(2048, activation='relu')(top_model)
    top_model= Dense(9, activation='relu')(top_model)
predictions=Dense(9, activation='softmax')(top_model)
model = Model(inputs=VGG_model.input, outputs=predictions)

           
              
opt = SGD(lr=0.001, momentum=0.9) #Use stochastic gradient descent. Also try others. 
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
##########################################
#Train the CNN model
history = model.fit(x_train, y_train_one_hot, epochs=100, validation_data = (x_test, y_test_one_hot))
            
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


accuracy = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, accuracy, 'y', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


prediction_NN = model.predict(x_test)
prediction_NN = np.argmax(prediction_NN, axis=-1)
prediction_NN = le.inverse_transform(prediction_NN)

#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, prediction_NN)
print(cm)
sns.heatmap(cm, annot=True)

#Check results on a few select images

#n=5 dog park. NN not as good as RF.
n=9  #Select the index of image to be loaded for testing
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
prediction = np.argmax(model.predict(input_img))  #argmax to convert categorical back to original
prediction = le.inverse_transform([prediction])  #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction)
print("The actual label for this image is: ", test_labels[n])



#Now, let us use features from convolutional network for RF
feature_extractor=model.predict(x_train)

features = feature_extractor.reshape(feature_extractor.shape[0], -1)

X_for_RF = features #This is our X input to RF

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 1000, random_state = 42)

# Train the model on training data
RF_model.fit(X_for_RF, y_train) #For sklearn no one hot encoding

#Send test data through same feature extractor process
X_test_feature = model.predict(x_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

#Now predict using the trained RF model. 
prediction_RF = RF_model.predict(X_test_features)
#Inverse le transform to get original label back. 
prediction_RF = le.inverse_transform(prediction_RF)

#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, prediction_RF))

#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, prediction_RF)
#print(cm)
sns.heatmap(cm, annot=True)

#Check results on a few select images
n=np.random.randint(0, x_test.shape[0])
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_feature=model.predict(input_img)
input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
prediction_RF = RF_model.predict(input_img_features)[0] 
prediction_RF = le.inverse_transform([prediction_RF])  #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction_RF)
print("The actual label for this image is: ", test_labels[n])

#Trainable parameters will be 0

















################################################################################################
              

# defining ImageDataGenerators
# ... initialization for training
#train_datagen = ImageDataGenerator(fill_mode="reflect",
#                                   rotation_range=45,
 #                                  horizontal_flip=True,
#                                   vertical_flip=True,
#                                   preprocessing_function=preprocessing_image_rgb)
# ... initialization for validation
#test_datagen = ImageDataGenerator(preprocessing_function=preprocessing_image_rgb)
# ... definition for training
#train_generator = train_datagen.flow_from_directory(path_to_train,
#                                                    target_size=(224, 224),
#                                                    batch_size=batch_size,
#                                                    class_mode='categorical')
# just for information
#class_indices = train_generator.class_indices
#print(class_indices)

# ... definition for validation
#validation_generator = test_datagen.flow_from_directory(path_to_validation,
#                                                        target_size=(224, 224),
#                                                        batch_size=batch_size,
#                                                        class_mode='categorical')
#
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional layers
#for layer in base_model.layers:
#    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
#model.compile(optimizer='adadelta', loss='categorical_crossentropy',
#              metrics=['categorical_accuracy'])


# generate callback to save best model w.r.t val_categorical_accuracy
if use_vgg:
    file_name = "vgg"
else:
    file_name = "dense"

checkpointer = ModelCheckpoint("../UCMerced_LandUses/dataset4%/" + file_name +
                               "_rgb_transfer_init." +
                               "{epoch:02d}-{val_categorical_accuracy:.3f}." +
                               "hdf5",
                               monitor='val_categorical_accuracy',
                               verbose=1,
                               save_best_only=True,
                               mode='max')

earlystopper = EarlyStopping(monitor='val_categorical_accuracy',
                             patience=21,
                             mode='max',
                             restore_best_weights=True)

tensorboard = TensorBoard(log_dir='./logs', write_graph=True, write_grads=True,
                          write_images=True, update_freq='epoch')

history = model.fit_generator(train_generator,
                              steps_per_epoch=100,
                              epochs=1000,
                              callbacks=[checkpointer, earlystopper,
                                         tensorboard],
                              validation_data=validation_generator,
                              validation_steps=50)
initial_epoch = len(history.history['loss'])+1
# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
names = []
for i, layer in enumerate(model.layers):
    names.append([i, layer.name, layer.trainable])
print(names)

if use_vgg:
    # we will freaze the first convolutional block and train all
    # remaining blocks, including top layers.
    for layer in model.layers[:4]:
        layer.trainable = False
    for layer in model.layers[4:]:
        layer.trainable = True
else:
    for layer in model.layers[:7]:
        layer.trainable = False
    for layer in model.layers[7:]:
        layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])


# generate callback to save best model w.r.t val_categorical_accuracy
if use_vgg:
    file_name = "vgg"
else:
    file_name = "dense"
checkpointer = ModelCheckpoint("../UCMerced_LandUses/dataset4%/" + file_name +
                               "_rgb_transfer_final." +
                               "{epoch:02d}-{val_categorical_accuracy:.3f}" +
                               ".hdf5",
                               monitor='val_categorical_accuracy',
                               verbose=1,
                               save_best_only=True,
                               mode='max')
earlystopper = EarlyStopping(monitor='val_categorical_accuracy',
                             patience=50,
                             mode='max')
model.fit_generator(train_generator,
                    steps_per_epoch=1000,
                    epochs=10000,
                    callbacks=[checkpointer, earlystopper, tensorboard],
                    validation_data=validation_generator,
                    validation_steps=500,
                    initial_epoch=initial_epoch)






































