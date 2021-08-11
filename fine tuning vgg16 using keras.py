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
import seaborn as sns
import keras
import keras_applications
import tensorflow as tf
import tensorboard
import cv2
from PIL import Image 

from tensorflow.python.platform import gfile
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import GlobalAveragePooling2D, Dense
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet201 as DenseNet

# Read input images and assign labels based on folder names
print(os.listdir("UCMerced_LandUses/dataset3%"))

SIZE = 224  #Resize images

#Capture training data and labels into respective lists
train_images = []
train_labels = [] 

for directory_path in glob.glob("UCMerced_LandUses/dataset3%/train/*"):
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
for directory_path in glob.glob("UCMerced_LandUses/dataset3%/validation/*"):
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
#Load model wothout classifier/fully connected layers
# parameters for CNN
use_vgg = False
if use_vgg:
    base_model = VGG(include_top=False, weights='imagenet', input_shape=(SIZE, SIZE, 3))
else:
    base_model = DenseNet(include_top=False, weights='imagenet', input_shape=(SIZE, SIZE, 3))
# add a global spatial average pooling layer
top_model = base_model.output
top_model = GlobalAveragePooling2D()(top_model)
# or just flatten the layers
#top_model = Flatten()(top_model) #?
# let's add a fully-connected layer
if use_vgg:
    # only in VGG19 a fully connected nn is added for classfication
    # DenseNet tends to overfitting if using additionally dense layers
    top_model = Dense(2048, activation='relu')(top_model)
    top_model = Dense(2048, activation='relu')(top_model)
# and a logistic layer
predictions = Dense(21, activation='softmax')(top_model)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
# print network structure
model.summary()


#defining  imageDataGenerators
#...... initialization for training 
train_datagen = ImageDataGenerator(fill_mode="reflect",
                                   rotation_range=45,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   preprocessing_function=preprocessing_image_rgb)
# ... initialization for validation
test_datagen = ImageDataGenerator(preprocessing_function=preprocessing_image_rgb)
# ... definition for training
train_generator = train_datagen.flow_from_directory(path_to_train,
                                                    target_size=(64, 64),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
#just for information 
#class_indices = train_generator.class_indices

                                                    

print(class_indices)
print(train_generator.class_indices)

# ... definition for validation
validation_generator = test_datagen.flow_from_directory(path_to_validation,
                                                        target_size=(64, 64),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
if use_vgg:
    file_name = "vgg"
else:
    file_name = "dense"

checkpointer = ModelCheckpoint("../data/models/" + file_name +
                               "_rgb_from_scratch." +
                               "{epoch:02d}--{val_categorical_accuracy:.3f}" +
                               ".hdf5", 
                               monitor='val_categorical_accuracy',
                               verbose=1,
                               save_best_only=True,
                               mode= 'max')
early_stopper = EarlyStopping(monitor='val_categorical_accuracy', 
                              patience=500,
                              mode='max')
model.fit_generator(train_generator,
                    steps_per_epoch=100,
                    epochs=100,
                    callbacks=[checkpointer, early_stopper],
                    validation_data=50)


#plot the training and validation accuracy and loss at each epoch
loss = model.history['loss']
val_loss = model.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = model.history['acc']
val_acc = model.history['val_acc']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
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
n=200 #Select the index of image to be loaded for testing
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
RF_model = RandomForestClassifier(n_estimators = 200, random_state = 42)

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
input_img_feature= model.predict(input_img)
input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
prediction_RF = RF_model.predict(input_img_features)[0] 
prediction_RF = le.inverse_transform([prediction_RF])  #Reverse the label encoder to original name
print("The prediction for this image is: ", prediction_RF)
print("The actual label for this image is: ", test_labels[n])



def plot_confusion_matrix(y_true, y_pred, matrix_title):
    """confusion matrix computation and display"""
    plt.figure(figsize=(9, 9), dpi=100)

    # use sklearn confusion matrix
    cm_array = confusion_matrix(y_true, y_pred)
    plt.imshow(cm_array[:-1, :-1], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(matrix_title, fontsize=16)

    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Number of images', rotation=270, labelpad=21, fontsize=12)

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))

    plt.xticks(xtick_marks, true_labels, rotation=90)
    plt.yticks(ytick_marks, pred_labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.tight_layout()

    plt.show()


# Classifier performance


def run_classifier(clfr, x_train_data, y_train_data, x_test_data, y_test_data, acc_str, matrix_header_str):
    """run chosen classifier and display results"""
    start_time = time.time()
    clfr.fit(x_train_data, y_train_data)
    y_pred = clfr.predict(x_test_data)
    print("%f seconds" % (time.time() - start_time))

    # confusion matrix computation and display
    print(acc_str.format(accuracy_score(y_test_data, y_pred) * 100))
    plot_confusion_matrix(y_test_data, y_pred, matrix_header_str)


# Read in images and extract features


# get images - labels are from the subdirectory names
if os.path.exists('features'):
    print('Pre-extracted features and labels found. Loading them ...')
    features = pickle.load(open('features', 'rb'))
    labels = pickle.load(open('labels', 'rb'))
else:
    print('No pre-extracted features - extracting features ...')
    # get the images and the labels from the sub-directory names
    dir_list = [x[0] for x in os.walk(images_dir)]
    dir_list = dir_list[1:]
    list_images = []
    for image_sub_dir in dir_list:
        sub_dir_images = [image_sub_dir + '/' + f for f in os.listdir(image_sub_dir) if re.search('tif|TIF', f)]
        list_images.extend(sub_dir_images)

    # extract features
    features, labels = extract_features(list_images)

    # save, so they can be used without re-running the last step which can be quite long
    pickle.dump(features, open('features', 'wb'))
    pickle.dump(labels, open('labels', 'wb'))
    print('CNN features obtained and saved.')



# Classification


# TSNE defaults:
# n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
# n_iter_without_progress=300, min_grad_norm=1e-07, metric=’euclidean’, init=’random’, verbose=0,
# random_state=None, method=’barnes_hut’, angle=0.5

# t-sne feature plot
if os.path.exists('tsne_features.npz'):
    print('t-sne features found. Loading ...')
    tsne_features = np.load('tsne_features.npz')['tsne_features']
else:
    print('No t-sne features found. Obtaining ...')
    tsne_features = TSNE().fit_transform(features)
    np.savez('tsne_features', tsne_features=tsne_features)
    print('t-sne features obtained and saved.')

plot_features(labels, tsne_features)

# prepare training and test datasets
X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.2, random_state=42)


# LinearSVC defaults:
# penalty=’l2’, loss=’squared_hinge’, dual=True, tol=0.0001, C=1.0, multi_class=’ovr’, fit_intercept=True,
# intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000

# classify the images with a Linear Support Vector Machine (SVM)
print('Support Vector Machine starting ...')
clf = LinearSVC()
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-SVM Accuracy: {0:0.1f}%", "SVM Confusion matrix")


# RandomForestClassifier/ExtraTreesClassifier defaults:
# (n_estimators=10, criterion='gini’, max_depth=None, min_samples_split=2, min_samples_leaf=1,
# min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0,
# min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False,
# class_weight=None)

# classify the images with a Extra Trees Classifier
print('Extra Trees Classifier starting ...')
clf = ExtraTreesClassifier(n_jobs=4,  n_estimators=100, criterion='gini', min_samples_split=10,
                           max_features=50, max_depth=40, min_samples_leaf=4)
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-ET Accuracy: {0:0.1f}%", "Extra Trees Confusion matrix")

# classify the images with a Random Forest Classifier
print('Random Forest Classifier starting ...')
clf = RandomForestClassifier(n_jobs=4, criterion='entropy', n_estimators=70, min_samples_split=5)
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-RF Accuracy: {0:0.1f}%", "Random Forest Confusion matrix")


# KNeighborsClassifier defaults:
# n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’, metric_params=None,
# n_jobs=1, **kwargs

# classify the images with a k-Nearest Neighbors Classifier
print('K-Nearest Neighbours Classifier starting ...')
clf = KNeighborsClassifier(n_neighbors=1, n_jobs=4)
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-KNN Accuracy: {0:0.1f}%",
               "K-Nearest Neighbor Confusion matrix")


# MPLClassifier defaults:
# hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’, alpha=0.0001, batch_size=’auto’,
# learning_rate=’constant’, learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None,
# tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
# validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08

# classify the image with a Multi-layer Perceptron Classifier
print('Multi-layer Perceptron Classifier starting ...')
clf = MLPClassifier()
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-MLP Accuracy: {0:0.1f}%",
               "Multi-layer Perceptron Confusion matrix")

# GaussianNB defaults:
# priors=None

# classify the images with a Gaussian Naive Bayes Classifier
print('Gaussian Naive Bayes Classifier starting ...')
clf = GaussianNB()
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-GNB Accuracy: {0:0.1f}%",
               "Gaussian Naive Bayes Confusion matrix")

# LinearDiscriminantAnalysis defaults:
# solver=’svd’, shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001

# classify the images with a Quadratic Discriminant Analysis Classifier
print('Linear Discriminant Analysis Classifier starting ...')
clf = LinearDiscriminantAnalysis()
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-LDA Accuracy: {0:0.1f}%",
               "Linear Discriminant Analysis Confusion matrix")

# QuadraticDiscriminantAnalysis defaults:
# priors=None, reg_param=0.0, store_covariance=False, tol=0.0001, store_covariances=None

# classify the images with a Quadratic Discriminant Analysis Classifier
print('Quadratic Discriminant Analysis Classifier starting ...')
clf = QuadraticDiscriminantAnalysis()
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-QDA Accuracy: {0:0.1f}%",
               "Quadratic Discriminant Analysis Confusion matrix")



