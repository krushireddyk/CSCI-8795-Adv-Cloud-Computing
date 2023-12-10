import requests
from google.cloud import storage
import numpy as np
import pandas as pd
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler

storage_client = storage.Client()
bucket = storage_client.get_bucket('valiant-ocean-401219')
predict_weather = bucket.blob('random_forest_model/random_forest.pkl')
predict_weather.download_to_filename('/tmp/random_forest.pkl')

#load saved model
with open('/tmp/random_forest.pkl', 'rb') as f:
  serverless_weather_prediction = pickle.load(f)
def predict(data):
  data_frame = pd.DataFrame(data)
  predictions = serverless_weather_prediction.predict(data_frame)
  # Convert NumPy array to Pandas DataFrame
  predictions_df = pd.DataFrame({'predictions': predictions})

  # Convert Pandas DataFrame to JSON string
  json_preds = predictions_df.to_json(orient='records')
  return {"predictions": json_preds, "success": 200}

def random_forest_predict(request):
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
