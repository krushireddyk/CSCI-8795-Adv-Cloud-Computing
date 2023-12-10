import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from google.cloud import storage
from sklearn.metrics import accuracy_score
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.optimizers import Adam

storage_client = storage.Client()
bucket = storage_client.get_bucket('cloud-computing-project-401219')
predict_weather = bucket.blob('01/01/data/weatherHistory.csv')
predict_weather.download_to_filename('/tmp/weatherHistory.csv')        
with open('/tmp/weatherHistory.csv','rb') as f:
    weather_dataset = pd.read_csv(f)

df = weather_dataset.dropna()
def encode_features(X):
    X['Summary'] = X['Summary'].astype('category')
    encoded_df_1 = pd.get_dummies(X['Summary'])
    encoded1 = pd.concat([X,encoded_df_1],axis=1)
    df_new=pd.concat([encoded1],axis=1)
    coded_df = df_new.drop(['Summary'],axis=1)
    return coded_df

def assignUniqueNumber(df):
    map = {'rain': 0, 'snow': 1}
    labelLowered = df["Precip_Type"].values
    labelNum = []
    for label in labelLowered:
      labelNum.append(map[label])
    df.loc[:, "label"] = labelNum
    return df

def lstm_weather_prediction_training_without_undersampling(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    df1 = assignUniqueNumber(df)
    y = df1.label.values
    X = df1.drop('Precip_Type', axis=1)
    X = X.drop('label', axis=1)
    X = X.drop('Formatted_Date', axis=1)
    X = X.drop('Daily_Summary', axis=1)
    X = X.drop('Loud_Cover', axis=1)
    X = encode_features(X)
    X = X.astype(float)
    y = y.astype(int)
    normalised_df_X=pd.DataFrame(X,columns=X.columns)
    print(normalised_df_X.columns)
    print(normalised_df_X)
    normalised_df_X = normalised_df_X.values
    normalised_df_X = np.reshape(normalised_df_X, (normalised_df_X.shape[0], 1, normalised_df_X.shape[1]))
    print(normalised_df_X)
    X_train, X_test, Y_train, Y_test = train_test_split(normalised_df_X,y, test_size = 0.10, random_state = 42)
    print(X_train.shape,Y_train.shape)
    print(X_test.shape,Y_test.shape)
    model = Sequential()
    model.add(LSTM(32, input_shape=(1, 34), dropout=0.1, recurrent_dropout=0, recurrent_activation="sigmoid", use_bias=True, unroll=False))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=0.005)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    epochs = 10
    batch_size = 16

    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1)
    print("test evaluate")
    print(X_test[0])
    accr = model.evaluate(X_test,Y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    preds1 = model.predict(X_test)
    print("predictions")
    print(preds1[0])
    
    # create an iterator object with write permission - model.pkl
    with open('/tmp/LSTM_Model_without_undersampling.pkl', 'wb') as files:
        pickle.dump(model, files)
    files.close()
    svm_model_trained = bucket.blob('LSTM_Model/LSTM_Model_without_undersampling.pkl')
    svm_model_trained.upload_from_filename('/tmp/LSTM_Model_without_undersampling.pkl')
    buckets = list(storage_client.list_buckets())
    print(buckets)
    return "success with accuracy: " + str(accr)
