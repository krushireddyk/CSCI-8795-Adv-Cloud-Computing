from google.cloud import storage
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

storage_client = storage.Client()
bucket = storage_client.get_bucket('cloud-computing-project-401219')
predict_weather = bucket.blob('LSTM_Model/LSTM_Model_without_undersampling.pkl')
predict_weather.download_to_filename('/tmp/weather_prediction_lstm.pkl')

def predict(data):
  threshold = 0.5
  mappings = {0:"rain",1:"snow"}
  data_frame = pd.DataFrame(data)
  print(data_frame)
  normalised_df_X = data_frame.values
  print(normalised_df_X)
  normalised_df_X1 = np.reshape(normalised_df_X, (normalised_df_X.shape[0], 1, normalised_df_X.shape[1]))
  print(normalised_df_X1)
  print("before prediction1")

  #load saved model
  with open('/tmp/weather_prediction_lstm.pkl', 'rb') as f:
    serverless_weather_prediction = pickle.load(f)

  predictions = serverless_weather_prediction.predict(normalised_df_X1)
  print("after prediction")
  print(predictions)
  preds3=[ mappings[1] if i >= threshold else mappings[0] for i in predictions  ]
  # Convert NumPy array to Pandas DataFrame
  predictions_df = pd.DataFrame({'predictions': preds3})

  # Convert Pandas DataFrame to JSON string
  json_preds = predictions_df.to_json(orient='records')
  return {"predictions": json_preds, "success": 200}

def lstm_prediction_GCP(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    request_json = request.get_json()
    data = request.get_json()
    predictions = predict(data)
    return predictions
