# Semi-Supervised Weather Image Classification

This project demonstrates a semi-supervised learning approach for classifying weather images into four categories: cloudy, rain, shine, and sunrise. The classification is performed using a Convolutional Neural Network (CNN) implemented in TensorFlow and Keras. The model is first trained on a labeled dataset and then improved with self-training on an unlabeled dataset.

And, in this attempt, we are using the same data and model structure as our previous supervised learning model, which can be found at [here](https://github.com/Potassium-chromate/CNN-for-recognizer-weather). The only difference is that we will be labeling only one-fifth of the data, while the rest will be left unlabeled.

## 1. Purposes
We are attempting to train a model using self-training with only one-fifth of the data used in supervised learning, while still achieving comparable performance to our previous training.

## 2.Target
We use CNN to to classify the type of weather present in the image.There will be a labeled training set, unlabeled training set and test set. Labeled set include 211 RGB pictures and a few gray picture. And the Unabeled set include 837 RGB pictures. The training set include 75 RGB pictures.
### One-hot encoding for the label  
`0:cloudy
1:rain
2:shine
3:sunrise`

## 3. Dependencies

- Python 3.6 or higher
- TensorFlow 2.0 or higher
- Keras
- Pillow
- NumPy
- scikit-learn
- seaborn
- pandas
- matplotlib

## 4.Method
### CNN structure
![Alt Text](https://github.com/Potassium-chromate/CNN-for-recognizer-weather/blob/main/Picture/Model%20structure.png)

### Process
1. Load labeled pictures and resize to (300,300,3)
2. Additional augmentation for the labeled pictures  
   `rotate 45 and 60 degrees`  
   `flipped`
3. Randomized the order of pictures and labels by using `sklearn.utils.shuffle`
4. Train the model using only the labeled images with augmentation.  
5. Begin self-training by using the remaining unlabeled images, with augmentation.
6. Every iteration should randomly choose 20% amount of data from last iteration. And combine with all of `x_pseudo_labeled` as the training data in this iteration.  
`subset_X_train = self.X_train[subset_indices]` 20% amount of data from last iteration  
`mix_X_train = np.concatenate((subset_X_train, x_pseudo_labeled))` combime the data randomly choose from last iteration with pseudo_labeled data.  
7. Generate a chart to evaluate the performance of the model  

## 5. Usage

1. Clone the repository to your local machine.

```bash
git clone https://github.com/yourusername/semi-supervised-weather-classification.git  
```
2. Change the paths in the Python script to your local paths for the labeled training, test, and unlabeled datasets. You should have the following folder structure:  
├── train(labeled)  
│   ├── cloudy  
│   ├── rain  
│   ├── shine  
│   └── sunrise  
├── test  
│   ├── cloudy  
│   ├── rain  
│   ├── shine  
│   └── sunrise  
└── mix(unlabeled)    
3. Run the script:
```
python semi_supervised_weather_classification.py
```
The script will train the CNN model using the labeled training dataset and then perform self-training on the unlabeled dataset. After the training is complete, the model's performance will be evaluated on the test dataset, and the results will be saved in an Excel file named "Result.xlsx". The trained model will be saved as "my_model.h5".

## 6. Result
### Before self-training  
`loss: 0.3927 - acc: 0.8489 - val_loss: 0.7928 - val_acc: 0.8133`
#### Accuracy curve
![Alt Text](https://github.com/Potassium-chromate/Semi-Supervised-Weather-Image-Classification/blob/main/picture/before%20self_training/accuracy.png)
#### Loss curve
![Alt Text](https://github.com/Potassium-chromate/Semi-Supervised-Weather-Image-Classification/blob/main/picture/before%20self_training/loss.png)
#### Train confusion
![Alt Text](https://github.com/Potassium-chromate/Semi-Supervised-Weather-Image-Classification/blob/main/picture/before%20self_training/Train%20Confusion%20Matrix.png)
#### Test confusion
![Alt Text](https://github.com/Potassium-chromate/Semi-Supervised-Weather-Image-Classification/blob/main/picture/before%20self_training/Test%20Confusion%20Matrix.png)  

### After self-training  
`loss: 0.1290 - acc: 0.9620 - val_loss: 1.5775 - val_acc: 0.8800`  
#### Accuracy curve
![Alt Text](https://github.com/Potassium-chromate/Semi-Supervised-Weather-Image-Classification/blob/main/picture/after%20self_training/accuracy.png)
#### Loss curve
![Alt Text](https://github.com/Potassium-chromate/Semi-Supervised-Weather-Image-Classification/blob/main/picture/after%20self_training/loss.png)
#### Train confusion
![Alt Text](https://github.com/Potassium-chromate/Semi-Supervised-Weather-Image-Classification/blob/main/picture/after%20self_training/Train%20Confusion%20Matrix.png)
#### Test confusion
![Alt Text](https://github.com/Potassium-chromate/Semi-Supervised-Weather-Image-Classification/blob/main/picture/after%20self_training/Test%20Confusion%20Matrix.png)
