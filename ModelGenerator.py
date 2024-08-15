# Required libraries
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import IndicatorCalculator as IC
#import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler



sample_path = "manipulate.csv"

out_put_model = "my_trained_model_1m_normalized.pkl"
regression_sensitivity = 0

def GenerateModel(train_data, test_data):
    # Download historical data for XAU/USD (gold)
   #test_data = yf.download("GC=F", period="1mo", interval="5m")
    
    test_data = IC.IndicatorCalculator(test_data, "none")
    train_data = IC.IndicatorCalculator(train_data, "none")
    
    test_data_output = IC.DataManipulator(test_data)
    train_data_output = IC.DataManipulator(train_data)
    

    # ======================== generating Logistic Regression Model ========================
    
    scaler = MinMaxScaler()
    train_data_trasform = scaler.fit_transform(train_data[IC.input_to_model])
    test_data_transform = scaler.fit_transform(test_data[IC.input_to_model])

    # Train logistic regression model
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(train_data_trasform, train_data_output)

    # Evaluate model
    y_pred = model.predict(test_data_transform)
    accuracy = (y_pred == test_data_output).mean()
    y_pred_proba = model.predict_proba(test_data_transform)

    # manual offset
    for i in range(len(y_pred)):
        buy_prob = y_pred_proba[i][0] + regression_sensitivity
        neutral_prob = y_pred_proba[i][1] - 2 * regression_sensitivity
        sell_prob = y_pred_proba[i][2] + regression_sensitivity

        if ((neutral_prob > buy_prob) and (neutral_prob > sell_prob)):
            y_pred[i] = 0
            break
        elif buy_prob > sell_prob:
            y_pred[i] = 1
            break
        elif buy_prob < sell_prob:
            y_pred[i] = -1
            break
        y_pred[i] = 0
        break

    print(classification_report(test_data_output, y_pred))
    print(f"Logistic Regression Model Accuracy: {accuracy:.2f}")
    
    test_data['Singal'] = test_data_output
    test_data['Singal_predict_regress'] = y_pred
    test_data = test_data.iloc[-5760:, :]
    test_data.to_csv(sample_path, sep=",")
    print(IC.input_to_model)
    with open(out_put_model, 'wb') as file:
        pickle.dump(model, file)

    # ======================== generating LSTM Model ========================
    features = train_data[IC.input_to_model].values
    # Normalize features to [0, 1] range
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    sequence_length = 30
    X, y = [], []
    for i in range(len(scaled_features) - sequence_length):
        X.append(scaled_features[i : i + sequence_length])
        y.append(scaled_features[i + sequence_length])

    X, y = np.array(X), np.array(y)

    # repeat for test data
    features = test_data[IC.input_to_model].values
    # Normalize features to [0, 1] range
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    sequence_length = 30
    X_test, y_test = [], []
    for i in range(len(scaled_features) - sequence_length):
        X_test.append(scaled_features[i : i + sequence_length])
        y_test.append(scaled_features[i + sequence_length])

    X, y = np.array(X), np.array(y)

    model_LSTM = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    tf.keras.layers.Dense(1)  # Output dimension matches the number of features (buy, sell, do nothing)
    ])

    model_LSTM.compile(optimizer='adam', loss='mean_squared_error')
    
    model_LSTM.fit(X, y, shuffle = False, epochs=50, batch_size=32)
    loss = model.evaluate(X_test, y_test)
    print(f"LSTM test Loss: {loss:.4f}")


    test_data['Singal_predict_regress'] = y_pred
    test_data.to_csv(sample_path, sep=",")


