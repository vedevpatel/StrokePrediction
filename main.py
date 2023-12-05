# Uploading Files 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from google.colab import files

uploaded = files.upload()


# Data Preparation

file_name = 'healthcare-dataset-stroke-data.csv'

dataframe = pd.read_csv('healthcare-dataset-stroke-data.csv')

print(dataframe.head())

median_bmi = dataframe['bmi'].median()
dataframe['bmi'].fillna(median_bmi, inplace=True)
print(dataframe.isnull().sum())

# Encoding Categorial Variables

dataframe_encoded = pd.get_dummies(dataframe, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])

x = dataframe_encoded.drop(columns=['stroke'])
y = dataframe_encoded['stroke']

# Handling Class Imbalance

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=100)
x_resampled, y_resampled = smote.fit_resample(x, y)

# Splitting Data

x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=100)

# Building Model

rf_model = RandomForestClassifier(random_state=100)
rf_model.fit(x_train, y_train)

# Making Predictions

y_prediction_rf = rf_model.predict(x_test)

# Evaluation

print("Random Forest Classifier:")
print(f'Accuracy: {accuracy_score(y_test, y_prediction_rf)}')
print(classification_report(y_test, y_prediction_rf))
