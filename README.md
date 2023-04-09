# Semi-Supervised Weather Image Classification

This project demonstrates a semi-supervised learning approach for classifying weather images into four categories: cloudy, rain, shine, and sunrise. The classification is performed using a Convolutional Neural Network (CNN) implemented in TensorFlow and Keras. The model is first trained on a labeled dataset and then improved with self-training on an unlabeled dataset.

## Dependencies

- Python 3.6 or higher
- TensorFlow 2.0 or higher
- Keras
- Pillow
- NumPy
- scikit-learn
- seaborn
- pandas
- matplotlib

## Usage

1. Clone the repository to your local machine.

```bash
git clone https://github.com/yourusername/semi-supervised-weather-classification.git  
```
2. Change the paths in the Python script to your local paths for the labeled training, test, and unlabeled datasets. You should have the following folder structure:  
├── train  
│   ├── cloudy  
│   ├── rain  
│   ├── shine  
│   └── sunrise  
├── test  
│   ├── cloudy  
│   ├── rain  
│   ├── shine  
│   └── sunrise  
└── mix  
3. Run the script:
```
python semi_supervised_weather_classification.py
```
The script will train the CNN model using the labeled training dataset and then perform self-training on the unlabeled dataset. After the training is complete, the model's performance will be evaluated on the test dataset, and the results will be saved in an Excel file named "Result.xlsx". The trained model will be saved as "my_model.h5".
