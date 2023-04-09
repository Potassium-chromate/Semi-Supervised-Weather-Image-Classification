# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:05:06 2023

@author: Eason
"""

import tensorflow as tf
from tensorflow.keras import layers
import os
from PIL import Image
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from PIL import ImageOps
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

class CNNAgent:
    # Define the model architecture
    def __init__(self, input_shape, train_data, train_label, test_data, test_label):
        self.shape = input_shape
        # Shuffle training data and labels
        self.X_train, self.Y_train = shuffle(train_data, train_label)
        self.X_test, self.Y_test = test_data, test_label
        # Create the CNN model
        self.model = self.create()
        self.history = None

    def create(self):
        # Build the CNN model
        model = tf.keras.Sequential([
            layers.Conv2D(32, (5, 5), activation='relu', input_shape=self.shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dense(4, activation='softmax')
        ])
        # Compile the model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        # Print the model summary
        model.summary()
        return model

    def train(self):
        # Train the CNN model and store the training history
        self.history = self.model.fit(self.X_train, self.Y_train, epochs=10, validation_data=(self.X_test, self.Y_test))
    
    def self_train(self, unlabeled_data, threshold=0.95, n_iterations=5):
       for iteration in range(n_iterations):
           # Predict on the unlabeled dataset
           predictions = self.model.predict(unlabeled_data)

           # Select instances with high confidence predictions
           pseudo_labeled_indices = np.where(predictions.max(axis=1) > threshold)[0]

           if len(pseudo_labeled_indices) == 0:
               print("No high-confidence pseudo-labels found. Stopping self-training.")
               break

           # Create pseudo-labeled dataset
           x_pseudo_labeled = unlabeled_data[pseudo_labeled_indices]
           y_pseudo_labeled = predictions[pseudo_labeled_indices].argmax(axis=1)
           y_pseudo_labeled_one_hot = tf.keras.utils.to_categorical(y_pseudo_labeled, num_classes=4)

           # Add the pseudo-labeled instances to the labeled dataset
           self.X_train = np.concatenate((self.X_train, x_pseudo_labeled))
           self.Y_train = np.concatenate((self.Y_train, y_pseudo_labeled_one_hot))

           # Remove the pseudo-labeled instances from the unlabeled dataset
           unlabeled_data = np.delete(unlabeled_data, pseudo_labeled_indices, axis=0)

           # Train the model using the updated labeled dataset
           self.train()
    
    def plot_curve(self):
        # Plot the training and validation accuracy and loss
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.show()

    def plot_confusion(self, true_label, predictions, title):
        # Plot the confusion matrix
        pre = np.argmax(predictions, axis=1)
        cm = confusion_matrix(np.argmax(true_label, axis=1), pre)
        fig = plt.figure(figsize=(8, 6))
        plt.title(title)
        sn.heatmap(cm, annot=True, cmap='OrRd', fmt='g')
        plt.xlabel('Predicted')
        plt.ylabel('True Label')
        plt.show()

    def result_csv(self, path, image_names, pre_label):
        # Save the prediction results to an Excel file
        result = pd.DataFrame({'Image Name': image_names, 'Predicted Category': pre_label})
        result.to_excel(path, index=False)

    def save(self, path):
        # Save the trained CNN model
        self.model.save(path)
        
        
class preprocessing:
    def __init__(self, train_path_list,test_path_list,target_size):
        self.target_size = target_size
        self.train_data , self.train_label ,self.train_name = self.load(train_path_list,arg = 'yes') #yes: additional augmentation
        self.test_data , self.test_label , self.test_name = self.load(test_path_list,arg = 'no') #no: no additional augmentation
        
        
    def load(self,path,arg = 'yes'):
        ret_data = []
        ret_label = []
        ret_name = []
        count = 0
        
        for i in path: #path is a list of folder path
            file_list = os.listdir(i)
            # Filter out non-image files (if any)
            img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            img_files = [f for f in file_list if os.path.splitext(f)[1].lower() in img_extensions]
            for f in img_files: #f is image name is the folder
                img_path = os.path.join(i, f)
                img = Image.open(img_path)
                img = img.convert('RGB')
                img = img.resize(self.target_size)
                
                # Original image
                img_array = np.array(img)
                label = np.array([0, 0, 0, 0])
                label[count] = 1 #Create one-hot encoding label
                ret_data.append(img_array)
                ret_label.append(label)
                ret_name.append(f)
                
                if arg == 'yes':
                    # Add rotated versions of the image
                    for angle in [45,60]:
                        rotated_img = img.rotate(angle)
                        rotated_img_array = np.array(rotated_img)
                        ret_data.append(rotated_img_array)
                        ret_label.append(label)
         
                    # Add flipped versions of the image
                    flipped_img = ImageOps.flip(img)
                    flipped_img_array = np.array(flipped_img)
                    ret_data.append(flipped_img_array)
                    ret_label.append(label)
                    
            count += 1    
        
        ret_data , ret_label = np.array(ret_data),np.array(ret_label)
        return ret_data,ret_label,ret_name
            
if __name__=='__main__':
    test_path = 'C:/Users/88696/Desktop/半監督式訓練集/weather_image_unlabeled/test'
    labeled_path = 'C:/Users/88696/Desktop/半監督式訓練集/weather_image_unlabeled/train'
    unlabeled_path = ['C:/Users/88696/Desktop/半監督式訓練集/weather_image_unlabeled/mix']
    
    label_train_path = [labeled_path+'/cloudy',labeled_path+'/rain',labeled_path+'/shine',labeled_path+'/sunrise']
    test_path = [test_path+'/cloudy',test_path+'/rain',test_path+'/shine',test_path+'/sunrise']
    target_size=(300,300)
    
    pre = preprocessing(label_train_path,test_path,target_size)
    
    # Load training data and labels
    train_data = pre.train_data/255
    train_label = pre.train_label
    train_name = pre.train_name
    # Load test data and labels
    test_data = pre.test_data/255
    test_label = pre.test_label
    test_name = pre.test_name
    
    unl_data,unl_label,unl_name = pre.load(unlabeled_path,arg = 'no')
    
    Agent = CNNAgent((300, 300, 3), train_data, train_label, test_data, test_label)
    Agent.train()
    Agent.self_train(unl_data / 255)  # Add self-training after the initial training
    Agent.plot_curve()
    Agent.save('C:/Users/88696/Desktop/半監督式訓練集/my_model.h5')
    
    #plot confusion
    train_pre = Agent.model.predict(train_data[::4,:,:,:])
    Agent.plot_confusion(train_label[::4],train_pre,"Train Confusion Matrix")
    test_pre = Agent.model.predict(test_data)
    Agent.plot_confusion(test_label,test_pre,"Test Confusion Matrix")
    #store predict result
    pre_label = np.argmax(test_pre, axis=1)
    Agent.result_csv('C:/Users/88696/Desktop/半監督式訓練集/Result.xlsx',test_name,pre_label)

