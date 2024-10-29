import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import PredictPipeline

from src.pipeline.predict_pipeline import CustomData

app = Flask(__name__)

#Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_datapoint', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('race_ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            writing_score = int(request.form.get('writing_score')),
            reading_score = int(request.form.get('reading_score'))
        )
        pref_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        predict_datapoint = predict_pipeline.predict(pref_df)
        return render_template('home.html', results=predict_datapoint)
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)