# Required libraries
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import IndicatorCalculator as IC
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler



sample_path = "manipulate.csv"

out_put_model = "my_trained_model_1m_normalized.pkl"
regression_sensitivity = 0.3

def GenerateModel(train_data, test_data):
    # Download historical data for XAU/USD (gold)
   #test_data = yf.download("GC=F", period="1mo", interval="5m")
    
    test_data = IC.IndicatorCalculator(test_data, "Real")
    train_data = IC.IndicatorCalculator(train_data, "Real")

    # manipulating train data
    #train_data["Signal"] = np.where(((train_data["EMA200"].shift(compare_period) - train_data["EMA200"])/train_data["EMA200"]) * 100 > signal_trigger,
    #    1,
    #    0,
    #)

    #train_data["Signal"] = np.where(((train_data["EMA200"].shift(compare_period) - train_data["EMA200"])/train_data["EMA200"]) * 100 < -(signal_trigger),
    #    -1,
    #    train_data["Signal"],
    #)

    ## manipulating test data
    #test_data["Signal"] = np.where(((test_data["EMA200"].shift(compare_period) - test_data["EMA200"])/test_data["EMA200"]) * 100 > signal_trigger,
    #    1,
    #    0,
    #)

    #test_data["Signal"] = np.where(((test_data["EMA200"].shift(compare_period) - test_data["EMA200"])/test_data["EMA200"]) * 100 < -(signal_trigger),
    #    -1,
    #    test_data["Signal"],
    #)
    test_data_output = IC.DataManipulator(test_data)
    train_data_output = IC.DataManipulator(train_data)

    #merge_signal = pd.Series(dtype="double")

    # manipulating data
    #if (len(buy_signal) == len(sell_signal)) and (
    #    len(buy_signal) == len(train_data["Signal"])
    #):
    #    for i in range(len(train_data["Signal"])):
    #        if (buy_signal[i] == 1) and (sell_signal[i] == 0):
    #            train_data["Signal"].values[i] = 1
    #            continue
    #        train_data["Signal"].values[i] = 0
            
    #    train_data.to_csv(sample_increase_path, sep=",")
    #    train_data["Signal"] = pd.Series(dtype="double")

    #    for i in range(len(train_data["Signal"])):
    #        if (sell_signal[i] == 2) and (buy_signal[i] == 0):
    #            train_data["Signal"].values[i] = 1
    #            continue
    #        train_data["Signal"].values[i] = 0
            
    #    train_data.to_csv(sample_decrease_path, sep=",")

    #train_data_increase = pd.read_csv(sample_increase_path)
    #train_data_decrease = pd.read_csv(sample_decrease_path)

    # ======================== generating Logistic Regression Model ========================

    # Split data into train and test sets
    scaler = MinMaxScaler()
    train_data_trasform = scaler.fit_transform(train_data[IC.input_to_model])
    test_data_transform = scaler.fit_transform(test_data[IC.input_to_model])

    #X = train_data_trasform
    #y = train_data_output
    #X_test = test_data_transform
    #y_test = test_data_output
    #X_train, X_test, y_train, y_test = train_test_split(
    #    Xi, yi, test_size=0.2, random_state=42
    #)

    # Train logistic regression model
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
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

    

    # decrease model
    #Xd = train_data_decrease[["EMA200","EMA50", "EMA2000", "RSI"]]
    #yd = train_data_decrease["Signal"]
    #Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    #    Xd, yd, test_size=0.2, random_state=42
    #)
    
    #model_d = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    #model_d.fit(Xd_train, yd_train)
    #yd_pred = model_d.predict(Xd_test)
    #accuracy_d = (model_d.predict(Xd_test) == yd_test).mean()
    #print(classification_report(yd_test, yd_pred))
    #print(f"Decrease Model Accuracy: {accuracy_d:.2f}")

    #with open(output_buy_model, 'wb') as file:
    #    pickle.dump(model_i, file
    test_data['Singal'] = test_data_output
    test_data['Singal_predict_regress'] = y_pred
    test_data.to_csv(sample_path, sep=",")
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


