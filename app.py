import joblib
from flask import Flask, render_template, redirect, url_for, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
# from flask_login import LoginManpercentage_of_time_with_abnormal_long_term_variabilityr, UserMixin, login_user, login_required, logout_user
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier




filename1 = 'model/fetal_health_risk.pkl'
model1 = joblib.load(filename1)



app = Flask(__name__, template_folder='templates')


##################################################################################
    
@app.route('/')
def index():
    return render_template("fetal.html")


@app.route("/fetal")
def fetal():
    return render_template("fetal.html")

##################################################################################

@app.route('/predictfetal', methods=['POST'])
def predictfetal():
    if request.method == 'POST':
        # Get form inputs
        percentage_of_time_with_abnormal_long_term_variability = float(request.form['percentage_of_time_with_abnormal_long_term_variability'])
        abnormal_short_term_variability = float(request.form['abnormal_short_term_variability'])
        mean_value_of_short_term_variability = float(request.form['mean_value_of_short_term_variability'])
        accelerations = float(request.form['accelerations'])
        prolongued_decelerations = float(request.form['prolongued_decelerations'])
        histogram_mode = float(request.form['histogram_mode'])
        uterine_contractions = float(request.form['uterine_contractions'])
        histogram_median = float(request.form['histogram_median'])
        histogram_mean = float(request.form['histogram_mean'])
       
              

        # Create a NumPy array with the input data
        data = np.array([[percentage_of_time_with_abnormal_long_term_variability, 
                          abnormal_short_term_variability, mean_value_of_short_term_variability, 
                          accelerations, prolongued_decelerations, 
                          histogram_mode, uterine_contractions, histogram_median, 
                          histogram_mean]])
        
        # Make prediction
        my_prediction = model1.predict(data)
        
        # Convert prediction to text
        prediction_text = ""
        if  my_prediction == 3:
            prediction_text = "Patient Health is Pathological"
        elif my_prediction == 2:
            prediction_text = "Patient Health is Suspicious"
        else:
            prediction_text = "Patient Health is Normal"


        return render_template('fetal_result.html', prediction_text=prediction_text)

##########################################################
########################

if __name__ == "__main__":
    app.run(debug=True)
