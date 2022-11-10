import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.model_selection import train_test_split
import numpy as np
import pickle


# Load the csv file
df = pd.read_csv("census.csv")

df['education_level']=df['education_level'].replace( [' 11th', ' 10th', ' 7th-8th',' 9th',
 ' 12th', ' 5th-6th', ' 1st-4th', ' Preschool'],' School')
 
#encoding
df['income'].replace({'>50K':1,'<=50K':0},inplace=True)
cat_cols=['workclass','education_level','marital-status', 'occupation', 'relationship', 'race', 'sex','native-country']

workclass_le=LabelEncoder()
df['workclass']=workclass_le.fit_transform(df['workclass'])

edu_lvl_le=LabelEncoder()
df['education_level']=edu_lvl_le.fit_transform(df['education_level'])

marital_le=LabelEncoder()
df['marital-status']=marital_le.fit_transform(df['marital-status'])

occ_le=LabelEncoder()
df['occupation']=occ_le.fit_transform(df['occupation'])

rel_le=LabelEncoder()
df['relationship']=rel_le.fit_transform(df['relationship'])

race_le=LabelEncoder()
df['race']=race_le.fit_transform(df['race'])

sex_le=LabelEncoder()
df['sex']=sex_le.fit_transform(df['sex'])

country_le=LabelEncoder()
df['native-country']=country_le.fit_transform(df['native-country'])


# Saving encoders
joblib.dump(workclass_le,'workclass_le.joblib',compress=9)
joblib.dump(edu_lvl_le,'edu_lvl_le.joblib',compress=9)
joblib.dump(marital_le,'marital_le.joblib',compress=9)
joblib.dump(occ_le,'occ_le.joblib',compress=9)
joblib.dump(rel_le,'rel_le.joblib',compress=9)
joblib.dump(race_le,'race_le.joblib',compress=9)
joblib.dump(sex_le,'sex_le.joblib',compress=9)
joblib.dump(country_le,'country_le.joblib',compress=9)


# Select independent and dependent variable
y = df['income']
x = df.drop(['income'], axis=1)

# Split the dataset into train and test
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size= 0.30 , random_state=42)

# Feature scaling
scaler = StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns = x.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns = x.columns)
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Instantiate the model
classifier = XGBClassifier(booster= 'dart',learning_rate= 0.1, max_depth= 8)

# Fit the model
classifier.fit(x_train , y_train)

# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))
print(classifier.score(x_test , y_test))