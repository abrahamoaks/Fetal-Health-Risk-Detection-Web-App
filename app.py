import joblib
from flask import Flask, render_template, request, jsonify
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField
from wtforms.validators import InputRequired, NumberRange
import numpy as np
from fhir.resources.diagnosticreport import DiagnosticReport
from fhir.resources.observation import Observation

# Load ML model
filename1 = 'model/fetal_health_risk.pkl'
model1 = joblib.load(filename1)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
Bootstrap(app)

# Define FlaskForm
class FetalHealthForm(FlaskForm):
    percentage_of_time_with_abnormal_long_term_variability = FloatField("Percentage of Abnormal Long-Term Variability", validators=[InputRequired()])
    abnormal_short_term_variability = FloatField("Abnormal Short-Term Variability", validators=[InputRequired()])
    mean_value_of_short_term_variability = FloatField("Mean Value of Short-Term Variability", validators=[InputRequired()])
    accelerations = FloatField("Accelerations", validators=[InputRequired()])
    prolongued_decelerations = FloatField("Prolongued Decelerations", validators=[InputRequired()])
    histogram_mode = FloatField("Histogram Mode", validators=[InputRequired()])
    uterine_contractions = FloatField("Uterine Contractions", validators=[InputRequired()])
    histogram_median = FloatField("Histogram Median", validators=[InputRequired()])
    histogram_mean = FloatField("Histogram Mean", validators=[InputRequired()])
    
    submit = SubmitField("Predict")

@app.route('/', methods=['GET', 'POST'])
def index():
    form = FetalHealthForm()
    prediction_text = None

    if form.validate_on_submit():
        # Collect form data
        data = np.array([[
            form.percentage_of_time_with_abnormal_long_term_variability.data,
            form.abnormal_short_term_variability.data,
            form.mean_value_of_short_term_variability.data,
            form.accelerations.data,
            form.prolongued_decelerations.data,
            form.histogram_mode.data,
            form.uterine_contractions.data,
            form.histogram_median.data,
            form.histogram_mean.data
        ]])

        # Make prediction
        my_prediction = model1.predict(data)[0]

        # Convert prediction to readable text
        prediction_text = "Pathological" if my_prediction == 3 else "Suspicious" if my_prediction == 2 else "Normal"

        # Convert to FHIR JSON
        diagnostic_report = DiagnosticReport.construct(
            status="final",
            code={"coding": [{"system": "http://loinc.org", "code": "57067-1", "display": "Fetal Health Assessment"}]},
            conclusion=prediction_text
        )

        return render_template('fetal_result.html', prediction_text=prediction_text, fhir_output=diagnostic_report.json())

    return render_template("fetal.html", form=form, prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
