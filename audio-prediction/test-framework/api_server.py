# Derived from: https://towardsdatascience.com/deploying-keras-deep-learning-models-with-flask-5da4181436a2
# Load libraries
import flask
from flask import Flask, Response, send_from_directory, request
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
from tensorflow.python.keras.backend import set_session
import time
import librosa
import os
import mysql.connector
import csv
import logging

# Instantiate flask
app = flask.Flask(__name__)

# Application config
app.config['UPLOAD_FOLDER'] = '/app/uploads'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['MYSQL'] = {
    "host": "mysql",
    "user": "audiouser",
    "passwd": "audiopass",
    "database": "audio",
    "auth_plugin": "mysql_native_password"
}
app.config['MODELS'] = {
    "classifer": {
        "path": "saved_models/weights.best.basic_cnn.hdf5",
        "labels": ['door', 'light', 'plate']
    }
}

# Connect to the database
def connection():
    db = mysql.connector.connect(
      host=app.config['MYSQL']['host'],
      user=app.config['MYSQL']['user'],
      passwd=app.config['MYSQL']['passwd'],
      database=app.config['MYSQL']['database'],
      auth_plugin=app.config['MYSQL']['auth_plugin']
    )

    return db

# we need to redefine our metric function in order
# to use it when loading the model
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return auc

# load the model, and pass in the custom metric function
global graph, session, model
graph = tf.compat.v1.get_default_graph()
session = tf.compat.v1.Session()
set_session(session)
model = load_model(app.config['MODELS']['classifer']['path'], custom_objects={'auc': auc})
print('Load weights...')
model.load_weights(app.config['MODELS']['classifer']['path'])
print('Weights loaded...')
model.make_predict_function()

# define a audio classification function as an endpoint
@app.route("/classify", methods=["GET", "POST"])
def classify_audio():
    db = connection()
    cursor = db.cursor()
    data = {"success": False}

    # Get the data
    original_label = request.args.get('label')
    file = request.files['file']
    temp_file = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    # If parameters are found, return a prediction
    if file:
        # Save the file in a temp folder
        file.save(temp_file)

        # Convert to an MFCC
        mfcc = extract_features(temp_file)
        if mfcc is None:
            app.logger.error("mfcc is empty")
            return "An error occurred."

        # Log it to the input table
        input_timestamp = str(time.time())
        mfcc_sql = np.array2string(mfcc, precision=2, separator=',', suppress_small=True)
        sql = "INSERT INTO input (label, mfcc, input_timestamp) VALUES (%s, %s, %s)"
        val = (original_label, mfcc_sql, input_timestamp)
        cursor.execute(sql, val)
        db.commit()

        # Run through audio classifer model
        with graph.as_default():
            with session.as_default():
                #data["label"] = model.predict(mfcc).to_list()
                # TODO: Change this to get the actual label name and probability from the model
                model_predict = model.predict(mfcc)[0]
                app.logger.info(model_predict)

                classifed_data = {}
                for key, value in enumerate(model_predict):
                    classifed_data[app.config['MODELS']['classifer']['labels'][key]] = value

                max_key = max(classifed_data, key=classifed_data.get)

                label_prediction = max_key
                label_probability = classifed_data[max_key]

                data["original_label"] = original_label
                data["classified_label"] = label_prediction
                data["classified_label_probability"] = str(label_probability.item())
                data["classified_all"] = str(classifed_data)
                data["classified_timestamp"] = str(time.time())
                data["success"] = True

        # Log it to the labels table
        sql = "INSERT INTO labels (input_label, classified_label, classified_label_probability, classified_all, classified_timestamp) VALUES (%s, %s, %s, %s, %s)"
        val = (original_label, data["classified_label"], data["classified_label_probability"], data["classified_all"], data["classified_timestamp"])
        cursor.execute(sql, val)
        db.commit()

        # Run the prediction model

        # Update the predictions table

        # Delete the temp file
        os.remove(temp_file)

    # Return a response in json format
    return flask.jsonify(data)

# GET the next prediction
@app.route("/predict", methods=["GET"])
def predict_location():
    db = connection()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM predictions")
    result = cursor.fetchall()

    data = {}
    for row in result:
        data[row[0]] = {
            'classified_label': row[2],
            'classified_label_probability': row[3],
            'next_location_label': row[4],
            'next_location_label_probability': row[5]
        }

    # Return a response in json format
    return flask.jsonify(data)

# GET a summary of the accuracies
@app.route("/summary", methods=["GET"])
def summary():
    # Vars
    data = {
        'classifier_average': 0.0,
        'prediction_average': 0.0
    }

    # Get the average classifier probability
    classifier_conn = connection()
    cursor_classifier = classifier_conn.cursor()
    cursor_classifier.execute("SELECT AVG(classified_label_probability) FROM labels")
    result_classifier = cursor_classifier.fetchone()
    app.logger.info(result_classifier)

    if result_classifier[0] is not None:
        data['classifier_average'] = result_classifier[0]

    cursor_classifier.close()

    # Get the average prediction probability
    predictions_conn = connection()
    cursor_predictions = predictions_conn.cursor()
    cursor_predictions.execute("SELECT AVG(next_location_label_probability) FROM predictions")
    result_predictions = cursor_predictions.fetchone()
    app.logger.info(result_predictions)

    if result_predictions[0] is not None:
        data['prediction_average'] = result_predictions[0]

    cursor_predictions.close()

    # Return a response in json format
    return flask.jsonify(data)

# Export all the database tables as CSV
@app.route("/export/<table_name>", methods=["GET"])
def export_data(table_name):
    db = connection()
    cursor = db.cursor()
    app.logger.info("Table name is "+table_name)
    cursor.execute("SELECT * FROM "+table_name)
    result = cursor.fetchall()

    filepath = "/app/downloads/"
    filename = table_name+".csv"
    full_file_path = filepath+filename
    app.logger.info("Writing file to "+full_file_path)

    # Clean up any previous export
    if os.path.exists(full_file_path):
        os.remove(full_file_path)

    # Set the CSV header row
    if table_name == 'input':
        csv_header = ['input_id', 'label', 'mfcc', 'input_timestamp']
    elif table_name == 'labels':
        csv_header = ['label_id', 'input_label', 'classified_label', 'classified_label_probability', 'classified_all' 'classified_timestamp']
    elif table_name == 'predictions':
        csv_header = ['prediction_id', 'label_id', 'classified_label', 'classified_label_probability', 'next_location_label', 'next_location_label_probability', 'prediction_timestamp']
    else:
        csv_header = []

    string_header = ','.join(csv_header)
    app.logger.info("Setting CSV header as "+string_header)

    # Write the table data to the CSV
    with open(full_file_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        writer.writeheader()
        for x in result:
            writer.writerow(x)

    if not os.path.exists(full_file_path):
        return 'Not found. Table '+table_name+' may not exist.'
    else:
        return send_from_directory(filepath, filename, as_attachment=True)

@app.route("/status", methods=["GET"])
def status():
    return "Cowabunga."

def extract_features(file_name):
    try:
        max_pad_length = 431
        audio, sample_rate = librosa.load(file_name)
        app.logger.info(sample_rate)

        n_mfcc = 120
        n_fft = 4096
        hop_length = 512
        n_mels = 512
        app.logger.info(sample_rate)

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        pad_width = max_pad_length-mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
        app.logger.info(mfccs.shape)
        #mfccsscaled = np.mean(mfccs.T,axis=0)

        # Reshape - this is different to the audio_classifer.py
        num_entries = 1
        num_rows = 120
        num_columns = 431
        num_channels = 1
        reshaped_mfccs = mfccs.reshape(num_entries, num_rows, num_columns, num_channels)
    except Exception as e:
        app.logger.error("Error encountered while parsing file: ", file_name, e)
        return None

    return reshaped_mfccs

# Start the flask app, allow remote connections
app.run(host='0.0.0.0')
