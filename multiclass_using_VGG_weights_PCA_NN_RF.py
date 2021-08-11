# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 21:54:06 2021

@author: bosulus
This code explains the process of using pretrained weights (VGG16) 
as feature extractors for both neural network and 
traditional machine learning (Random Forest). 
It also uses dimensional reduction using PCA to reduce the number of features for
speedy training.
"""
import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
from keras.layers.normalization import BatchNormalization
import os
import seaborn as sns
from keras.applications.vgg16 import VGG16
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.densenet import DenseNet201
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16 as VGG
from keras.applications.densenet import DenseNet201 as DenseNet
from keras.optimizers import SGD
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
#from image_functions import preprocessing_image_rgb 
from keras import backend as K


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

#############################
#Load VGG model with imagenet trained weights and without classifier/fully connected layers
#We will use this as feature extractor. 
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

#Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in VGG_model.layers:
	layer.trainable = False
    
VGG_model.summary()  #Trainable parameters will be 0


#Now, let us extract features using VGG imagenet weights
#Train features
train_feature_extractor=VGG_model.predict(x_train)
train_features = train_feature_extractor.reshape(train_feature_extractor.shape[0], -1)
#test features
test_feature_extractor=VGG_model.predict(x_test)
test_features = test_feature_extractor.reshape(test_feature_extractor.shape[0], -1)

# Reduce dimensions using PCA
from sklearn.decomposition import PCA

#First verfiy the ideal number of PCA components to not lose much information. 
#Try to retain 90% information, so look where the curve starts to flatten.
#Remember that the n_components must be lower than the number of rows or columns (features)
pca_test = PCA(n_components=1650) #
pca_test.fit(train_features)
plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cum variance")

#Pick the optimal number of components. This is how many features we will have 
#for our machine learning
n_PCA_components = 1650
pca = PCA(n_components=n_PCA_components)
train_PCA = pca.fit_transform(train_features)
test_PCA = pca.transform(test_features) #Make sure you are just transforming, not fitting. 

#If we want 90% information captured we can also try ...
#pca=PCA(0.9)
#principalComponents = pca.fit_transform(X_for_RF)

############## Neural Network Approach ##################

##Add hidden dense layers and final output/classifier layer.
model0 = Sequential()
inputs = Input(shape=(n_PCA_components,)) #Shape = n_components
hidden = Dense(256, activation='relu')(inputs)
#hidden1 = Dense(512, activation='relu')(inputs)
#hidden2 = Dense(256, activation='relu')(hidden1)
output = Dense(9, activation='softmax')(hidden)
model0 = Model(inputs=inputs, outputs=output)

print(model0.summary())
#
model0.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])

import datetime
start = datetime.datetime.now()
#Fit the model. Do not forget to use on-hot-encoded Y values. 
model0.fit(train_PCA, y_train_one_hot, epochs=20, verbose=1)

end = datetime.datetime.now()
print("Total execution time with PCA is: ", end-start)

##Predict on test dataset
predict_test = model0.predict(test_PCA)
predict_test = np.argmax(predict_test, axis=1)
predict_test = le.inverse_transform(predict_test)
#
#
##Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, predict_test))
model0.save('EAC_Dataset/save_models_2experiement/vgg16_pca_model0.hdf5')


#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, predict_test)
#print(cm)
sns.heatmap(cm, annot=True)

#Check results on a few select images
n=np.random.randint(0, x_test.shape[0])
img = x_test[n]
plt.imshow(img)

input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_feature=VGG_model.predict(input_img)
input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
input_img_PCA = pca.transform(input_img_features)
prediction_img = model0.predict(input_img_PCA)
prediction_img = np.argmax(prediction_img, axis=1)
prediction_img = le.inverse_transform(prediction_img)  #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction_img)
print("The actual label for this image is: ", test_labels[n])

############################################################
#RANDOM FOREST implementation (Uncomment to run this part)
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 16500, random_state = 42)
# Train the model on training data
RF_model.fit(train_PCA, y_train) #For sklearn no one hot encoding
#Send test data through same feature extractor process
#X_test_feature = VGG_model.predict(x_test)
#X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)
#Now predict using the trained RF model. 
prediction_RF = RF_model.predict(test_PCA)
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
#######################################################
#MODEL2 InceptionResNetV2

pre_trained_model = InceptionResNetV2( weights="imagenet", include_top=False, input_shape=(SIZE, SIZE, 3))
pre_trained_model.summary()
for layer in pre_trained_model.layers:
    layer.trainable = False 
pre_trained_model.summary()

#Now, let us extract features using InceptionResNetV2 imagenet weights
#Train features
train_feature_extractor=pre_trained_model.predict(x_train)
train_features = train_feature_extractor.reshape(train_feature_extractor.shape[0], -1)
#test features
test_feature_extractor=pre_trained_model.predict(x_test)
test_features = test_feature_extractor.reshape(test_feature_extractor.shape[0], -1)

# Reduce dimensions using PCA
from sklearn.decomposition import PCA

#First verfiy the ideal number of PCA components to not lose much information. 
#Try to retain 90% information, so look where the curve starts to flatten.
#Remember that the n_components must be lower than the number of rows or columns (features)
pca_test = PCA(n_components=1650) #
pca_test.fit(train_features)
plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cum variance")

#Pick the optimal number of components. This is how many features we will have 
#for our machine learning
n_PCA_components = 1650
pca = PCA(n_components=n_PCA_components)
train_PCA = pca.fit_transform(train_features)
test_PCA = pca.transform(test_features) #Make sure you are just transforming, not fitting. 

#If we want 90% information captured we can also try ...
#pca=PCA(0.9)
#principalComponents = pca.fit_transform(X_for_RF)

############## Neural Network Approach ##################

##Add hidden dense layers and final output/classifier layer.
model2 = Sequential()
hidden = Dense(256, activation='relu')(inputs)
#hidden1 = Dense(512, activation='relu')(inputs)
#hidden2 = Dense(256, activation='relu')(hidden1)
output = Dense(9, activation='softmax')(hidden)
model2 = Model(inputs=inputs, outputs=output)

inputs = Input(shape=(n_PCA_components,)) #Shape = n_components
print(model2.summary())
#
model2.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])

import datetime
start = datetime.datetime.now()
#Fit the model. Do not forget to use on-hot-encoded Y values. 
model2.fit(train_PCA, y_train_one_hot, epochs=20, verbose=1)

end = datetime.datetime.now()
print("Total execution time with PCA is: ", end-start)

##Predict on test dataset
predict_test = model2.predict(test_PCA)
predict_test = np.argmax(predict_test, axis=1)
predict_test = le.inverse_transform(predict_test)
#
#
##Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, predict_test))
model2.save('EAC_Dataset/save_models_2experiement/inceptionresenetv2_pca_mode2.hdf5')
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, predict_test)
#print(cm)
sns.heatmap(cm, annot=True)

#Check results on a few select images
n=np.random.randint(0, x_test.shape[0])
img = x_test[n]
plt.imshow(img)

input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_feature=pre_trained_model.predict(input_img)
input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
input_img_PCA = pca.transform(input_img_features)
prediction_img = model2.predict(input_img_PCA)
prediction_img = np.argmax(prediction_img, axis=1)
prediction_img = le.inverse_transform(prediction_img)  #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction_img)
print("The actual label for this image is: ", test_labels[n])
##############################################

#model3 DenseNet201

pre_trained_model = DenseNet201(input_shape=(SIZE, SIZE, 3), include_top=False, weights="imagenet")
pre_trained_model.summary()

for layer in pre_trained_model.layers:
    layer.trainable = False 
pre_trained_model.summary()

#Now, let us extract features using DenseNet201 imagenet weights
#Train features
train_feature_extractor=pre_trained_model.predict(x_train)
train_features = train_feature_extractor.reshape(train_feature_extractor.shape[0], -1)
#test features
test_feature_extractor=pre_trained_model.predict(x_test)
test_features = test_feature_extractor.reshape(test_feature_extractor.shape[0], -1)

# Reduce dimensions using PCA
from sklearn.decomposition import PCA

#First verfiy the ideal number of PCA components to not lose much information. 
#Try to retain 90% information, so look where the curve starts to flatten.
#Remember that the n_components must be lower than the number of rows or columns (features)
pca_test = PCA(n_components=1650) #
pca_test.fit(train_features)
plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cum variance")

#Pick the optimal number of components. This is how many features we will have 
#for our machine learning
n_PCA_components = 1650
pca = PCA(n_components=n_PCA_components)
train_PCA = pca.fit_transform(train_features)
test_PCA = pca.transform(test_features) #Make sure you are just transforming, not fitting. 

#If we want 90% information captured we can also try ...
#pca=PCA(0.9)
#principalComponents = pca.fit_transform(X_for_RF)

############## Neural Network Approach ##################

##Add hidden dense layers and final output/classifier layer.
model3 = Sequential()
inputs = Input(shape=(n_PCA_components,)) #Shape = n_components
hidden = Dense(256, activation='relu')(inputs)
#hidden1 = Dense(512, activation='relu')(inputs)
#hidden2 = Dense(256, activation='relu')(hidden1)
output = Dense(9, activation='softmax')(hidden)
model3 = Model(inputs=inputs, outputs=output)

print(model3.summary())
#
model3.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])

import datetime
start = datetime.datetime.now()
#Fit the model. Do not forget to use on-hot-encoded Y values. 
model3.fit(train_PCA, y_train_one_hot, epochs=20, verbose=1)

end = datetime.datetime.now()
print("Total execution time with PCA is: ", end-start)

##Predict on test dataset
predict_test = model3.predict(test_PCA)
predict_test = np.argmax(predict_test, axis=1)
predict_test = le.inverse_transform(predict_test)
#
#
##Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, predict_test))
model3.save('EAC_Dataset/save_models_2experiement/dense201_pca_model3.hdf5')
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, predict_test)
#print(cm)
sns.heatmap(cm, annot=True)

#Check results on a few select images
n=np.random.randint(0, x_test.shape[0])
img = x_test[n]
plt.imshow(img)

input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_feature=pre_trained_model.predict(input_img)
input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
input_img_PCA = pca.transform(input_img_features)
prediction_img = model3.predict(input_img_PCA)
prediction_img = np.argmax(prediction_img, axis=1)
prediction_img = le.inverse_transform(prediction_img)  #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction_img)
print("The actual label for this image is: ", test_labels[n])
###########################################

#model4 InceptionV3

pre_trained_model = InceptionV3(input_shape=(SIZE, SIZE, 3), include_top=False, weights="imagenet")


for layer in pre_trained_model.layers:
    layer.trainable=False 
pre_trained_model.summary()

#now, let use extract features using INCEPTIONV3 imagenet weight 
#train features 

train_feature_extractor=pre_trained_model.predict(x_train)
train_features = train_feature_extractor.reshape(train_feature_extractor.shape[0], -1)
#test features
test_feature_extractor=pre_trained_model.predict(x_test)
test_features = test_feature_extractor.reshape(test_feature_extractor.shape[0], -1)

# Reduce dimensions using PCA
from sklearn.decomposition import PCA

#First verfiy the ideal number of PCA components to not lose much information. 
#Try to retain 90% information, so look where the curve starts to flatten.
#Remember that the n_components must be lower than the number of rows or columns (features)
pca_test = PCA(n_components=1650) #
pca_test.fit(train_features)
plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cum variance")

#Pick the optimal number of components. This is how many features we will have 
#for our machine learning
n_PCA_components = 1650
pca = PCA(n_components=n_PCA_components)
train_PCA = pca.fit_transform(train_features)
test_PCA = pca.transform(test_features) #Make sure you are just transforming, not fitting. 


#If we want 90% information captured we can also try ...
#pca=PCA(0.9)
#principalComponents = pca.fit_transform(X_for_RF)

############## Neural Network Approach ##################

##Add hidden dense layers and final output/classifier layer.
model4 = Sequential()
inputs = Input(shape=(n_PCA_components,)) #Shape = n_components
hidden = Dense(256, activation='relu')(inputs)
#hidden1 = Dense(512, activation='relu')(inputs)
#hidden2 = Dense(256, activation='relu')(hidden1)
output = Dense(9, activation='softmax')(hidden)
model4 = Model(inputs=inputs, outputs=output)

print(model4.summary())


model4.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])

import datetime
start = datetime.datetime.now()
#Fit the model. Do not forget to use on-hot-encoded Y values. 
model4.fit(train_PCA, y_train_one_hot, epochs=20, verbose=1)

end = datetime.datetime.now()
print("Total execution time with PCA is: ", end-start)

##Predict on test dataset
predict_test = model4.predict(test_PCA)
predict_test = np.argmax(predict_test, axis=1)
predict_test = le.inverse_transform(predict_test)
#
#
##Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, predict_test))
model4.save('EAC_Dataset/save_models_2experiement/inceptionv3_pca_model4.hdf5')
#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, predict_test)
#print(cm)
sns.heatmap(cm, annot=True)

#Check results on a few select images
n=np.random.randint(0, x_test.shape[0])
img = x_test[n]
plt.imshow(img)

input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_feature=pre_trained_model.predict(input_img)
input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
input_img_PCA = pca.transform(input_img_features)
prediction_img = model4.predict(input_img_PCA)
prediction_img = np.argmax(prediction_img, axis=1)
prediction_img = le.inverse_transform(prediction_img)  #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction_img)
print("The actual label for this image is: ", test_labels[n])


#########################################################################

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
    top_model= Dense(256, activation='relu')(top_model)
predictions=Dense(9, activation='softmax')(top_model)
model = Model(inputs=VGG_model.input, outputs=predictions)

           
              
opt = SGD(lr=0.001, momentum=0.9) #Use stochastic gradient descent. Also try others. 
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()



### Model average / sum Ensemble
# Simple sum of all outputs / predictions and argmax across all classes
########


from keras.models import load_model
from sklearn.metrics import accuracy_score
#model0 = load_model('EAC_Dataset/save_models_2experiement/vgg16_pca_model0.hdf5')
model1 = load_model('EAC_Dataset/save_models_2experiement/inceptionresenetv2_pca_mode2.hdf5')
model2 = load_model('EAC_Dataset/save_models_2experiement/dense201_pca_model3.hdf5')
model3 = load_model('EAC_Dataset/save_models_2experiement/inceptionv3_pca_model4.hdf5')

models = [ model1, model2, model3]

preds = [model.predict(x_test) for model in models]
preds=np.array(preds)
summed = np.sum(preds, axis=0)

# argmax across classes
ensemble_prediction = np.argmax(summed, axis=1)

prediction1 = model1.predict_classes(x_test)
prediction2 = model2.predict_classes(x_test)
prediction3 = model3.predict_classes(x_test)

accuracy1 = accuracy_score(y_test, prediction1)
accuracy2 = accuracy_score(y_test, prediction2)
accuracy3 = accuracy_score(y_test, prediction3)
ensemble_accuracy = accuracy_score(y_test, ensemble_prediction)
print('Accuracy Score for model1 = ', accuracy1)
print('Accuracy Score for model2 = ', accuracy2)
print('Accuracy Score for model3 = ', accuracy3)
print('Accuracy Score for average ensemble = ', ensemble_accuracy)

########################################
#Weighted average ensemble
models = [model1, model2, model3]
preds = [model.predict(x_test) for model in models]
preds=np.array(preds)
weights = [0.4, 0.2, 0.4]

#Use tensordot to sum the products of all elements over specified axes.
weighted_preds = np.tensordot(preds, weights, axes=((0),(0)))
weighted_ensemble_prediction = np.argmax(weighted_preds, axis=1)

weighted_accuracy = accuracy_score(y_test, weighted_ensemble_prediction)

print('Accuracy Score for model1 = ', accuracy1)
print('Accuracy Score for model2 = ', accuracy2)
print('Accuracy Score for model3 = ', accuracy3)
print('Accuracy Score for average ensemble = ', ensemble_accuracy)
print('Accuracy Score for weighted average ensemble = ', weighted_accuracy)

########################################
#Grid search for the best combination of w1, w2, w3 that gives maximum acuracy
models = [model1, model2, model3]
preds1 = [model.predict(x_test) for model in models]
preds1=np.array(preds1)

import pandas as pd 
df = pd.DataFrame([])

for w1 in range(0, 5):
    for w2 in range(0,5):
        for w3 in range(0,5):
            wts = [w1/10.,w2/10.,w3/10.]
            wted_preds1 = np.tensordot(preds1, wts, axes=((0),(0)))
            wted_ensemble_pred = np.argmax(wted_preds1, axis=1)
            weighted_accuracy = accuracy_score(y_test, wted_ensemble_pred)
            df = df.append(pd.DataFrame({'wt1':wts[0],'wt2':wts[1], 
                                         'wt3':wts[2], 'acc':weighted_accuracy*100}, index=[0]), ignore_index=True)
            
max_acc_row = df.iloc[df['acc'].idxmax()]
print("Max accuracy of ", max_acc_row[0], " obained with w1=", max_acc_row[1],
      " w2=", max_acc_row[2], " and w3=", max_acc_row[3])         

###########################################################################
### Explore metrics for the ideal weighted ensemble model. 

models = [model1, model2, model3]
preds = [model.predict(x_test) for model in models]
preds=np.array(preds)
ideal_weights = [0.4, 0.1, 0.2] 

#Use tensordot to sum the products of all elements over specified axes.
ideal_weighted_preds = np.tensordot(preds, ideal_weights, axes=((0),(0)))
ideal_weighted_ensemble_prediction = np.argmax(ideal_weighted_preds, axis=1)

ideal_weighted_accuracy = accuracy_score(y_test, ideal_weighted_ensemble_prediction)



i = random.randint(1,len(ideal_weighted_ensemble_prediction))
plt.imshow(x_test[i,:,:,0]) 
print("Predicted Label: ", y_test_one_hot[int(ideal_weighted_ensemble_prediction[i])])
print("True Label: ", y_test_one_hot[int(y_test[i])])

from sklearn.metrics import confusion_matrix

#Print confusion matrix
cm = confusion_matrix(y_test, ideal_weighted_ensemble_prediction)

fig, ax = plt.subplots(figsize=(12,12))
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)


#PLot fractional incorrect misclassifications
incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
fig, ax = plt.subplots(figsize=(12,12))
plt.bar(np.arange(9), incorr_fraction)
plt.xlabel('True Label')
plt.ylabel('Fraction of incorrect predictions')
plt.xticks(np.arange(418), y_test_one_hot) 

