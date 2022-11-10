import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template
import pickle
import joblib


# Create flask app
flask_app = Flask(__name__)
#import model
model = pickle.load(open('model.pkl' , 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


#encoder
workclass_le=joblib.load('workclass_le.joblib')
edu_lvl_le=joblib.load('edu_lvl_le.joblib')
marital_le=joblib.load('marital_le.joblib')
occ_le=joblib.load('occ_le.joblib')
rel_le=joblib.load('rel_le.joblib')
race_le=joblib.load('race_le.joblib')
sex_le=joblib.load('sex_le.joblib')
country_le=joblib.load('country_le.joblib')

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():

    features = [' '+x for x in request.form.values()]
    encoded_features=[]
    
    features[1]=(workclass_le.transform([features[1]]))[0]
    features[2]=(edu_lvl_le.transform([features[2]]))[0]
    features[4]=(marital_le.transform([features[4]]))[0]
    features[5]=(occ_le.transform([features[5]]))[0]
    features[6]=(rel_le.transform([features[6]]))[0]
    features[7]=(race_le.transform([features[7]]))[0]
    features[8]=(sex_le.transform([features[8]]))[0]
    features[12]=(country_le.transform([features[12]]))[0]

    # encoded_features = [np.array(features)]

    df=pd.DataFrame(columns=['age', 'workclass', 'education_level', 'education-num',
       'marital-status', 'occupation', 'relationship', 'race', 'sex',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'])

    df = pd.concat([pd.DataFrame([features], columns=df.columns), df], ignore_index=True)
    df=df.astype('float')
    # scaler = StandardScaler()
    df = pd.DataFrame(scaler.transform(df), columns = df.columns)

    prediction = model.predict(df)

    return render_template("index.html", prediction_text = prediction )
    # if prediction==1:
    #     return render_template("index.html", prediction_text = "This person can be a donor for the charity")
    # else:
    #     return render_template("index.html", prediction_text = "Unfortunately ,this person can not be a donor for the charity")

if __name__ == "__main__":
    flask_app.run(debug=True)