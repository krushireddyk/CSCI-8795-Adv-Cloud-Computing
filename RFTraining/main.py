import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from google.cloud import storage
from sklearn.metrics import accuracy_score
import pickle

storage_client = storage.Client()
bucket = storage_client.get_bucket('valiant-ocean-401219')
predict_weather = bucket.blob('weatherHistoryKR.csv')
predict_weather.download_to_filename('/tmp/weatherHistoryKR.csv')
with open('/tmp/weatherHistoryKR.csv','rb') as f:
    weather_dataset = pd.read_csv(f)

df = weather_dataset.dropna()
balanced_df = df.drop('Formatted_Date', axis=1)
balanced_df = balanced_df.drop('Daily_Summary', axis=1)
def encode_features(X):
    X['Summary'] = X['Summary'].astype('category')
    encoded_df_1 = pd.get_dummies(X['Summary'])
    encoded1 = pd.concat([X,encoded_df_1],axis=1)
    df_new=pd.concat([encoded1],axis=1)
    coded_df = df_new.drop(['Summary'],axis=1)
    return coded_df
    
def random_forest_train(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    balanced_df1 = encode_features(balanced_df)
    print(balanced_df1.columns)
    # Data preprocessing
    # Assume 'RainOrSnow' is the target variable
    X = balanced_df1[['Temperature__C_', 'Apparent_Temperature__C_', 'Humidity', 'Wind_Speed__km_h_', 'Wind_Bearing__degrees_', 'Visibility__km_', 'Pressure__millibars_', 'Breezy', 'Breezy and Foggy', 'Breezy and Mostly Cloudy', 'Breezy and Overcast', 'Breezy and Partly Cloudy', 'Clear', 'Drizzle', 'Dry', 'Dry and Mostly Cloudy', 'Dry and Partly Cloudy', 'Foggy', 'Humid and Mostly Cloudy', 'Humid and Partly Cloudy', 'Light Rain', 'Mostly Cloudy', 'Overcast', 'Partly Cloudy', 'Windy', 'Windy and Foggy', 'Windy and Mostly Cloudy', 'Windy and Overcast', 'Windy and Partly Cloudy']]
    y = balanced_df1['Precip_Type']  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an SVM model (you can adjust parameters like kernel, C, gamma, etc.)
    #model = SVC(kernel='linear')
    model = RandomForestClassifier()

    # Train the logistic regression model
    model.fit(X_train, y_train)
    # Make predictions on the test set
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    correct =0
    incorrect = 0
    sub = X_test
    sub["actual"] = y_test
    sub["predicted"] = predictions
    for index, row in sub.iterrows():
        if row["predicted"] == row["actual"]:
            correct += 1
        else:
            incorrect +=1
    print(correct/(correct+incorrect)*100)
    # create an iterator object with write permission - model.pkl
    with open('/tmp/random_forest.pkl', 'wb') as files:
        pickle.dump(model, files)
    files.close()
    svm_model_trained = bucket.blob('random_forest_model/random_forest.pkl')
    svm_model_trained.upload_from_filename('/tmp/random_forest.pkl')
    buckets = list(storage_client.list_buckets())
    print(buckets)
    return "success with accuracy: " + str(accuracy)