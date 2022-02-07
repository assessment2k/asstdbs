from flask import Flask
import requests
import json
import joblib
from flask import Flask, request, jsonify

import joblib
import os
import pandas as pd

data_dir = '.\data'

datasrc = lambda x: os.path.join(data_dir, x)
df_labels = pd.read_csv(datasrc('labels.csv'))


df_labels.genre = pd.Categorical(df_labels.genre)
df_labels['label'] = df_labels.genre.cat.codes
labels = dict(enumerate(df_labels.genre.cat.categories))

def predict_genre(model, data_to_predict):

    y_pred = model.predict(data_to_predict)
    return labels[y_pred[0]]


app = Flask('app')

@app.route('/test', methods=['GET'])

def test():
    return 'Ping App'
joblib.load

@app.route('/predict', methods=['POST'])
def predict():
    data_to_predict = request.get_json()
#     print(data_to_predict)
    with open('./random_forest.joblib', 'rb') as f_in:
        model = joblib.load(f_in)
        f_in.close()
    predictions = predict_genre(model, data_to_predict)

    result = predictions
    return result

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)