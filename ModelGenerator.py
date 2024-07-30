# Required libraries
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import IndicatorCalculator as IC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

signal_trigger = 0.2 # percentage of price change
compare_period = -10

sample_path = "manipulate.csv"

out_put_model = "my_trained_model.pkl"

def GenerateModel(train_data):
    # Download historical data for XAU/USD (gold)
    test_data = yf.download("GC=F", period="1mo", interval="5m")
    
    test_data = IC.IndicatorCalculator(test_data, "Test")
    train_data = IC.IndicatorCalculator(train_data, "Real")

    # manipulating train data
    train_data["Signal"] = np.where(((train_data["EMA20"].shift(compare_period) - train_data["EMA20"])/train_data["EMA20"]) * 100 > signal_trigger,
        1,
        0,
    )

    train_data["Signal"] = np.where(((train_data["EMA20"].shift(compare_period) - train_data["EMA20"])/train_data["EMA20"]) * 100 < -(signal_trigger),
        -1,
        train_data["Signal"],
    )

    # manipulating test data
    test_data["Signal"] = np.where(((test_data["EMA20"].shift(compare_period) - test_data["EMA20"])/test_data["EMA20"]) * 100 > signal_trigger,
        1,
        0,
    )

    test_data["Signal"] = np.where(((test_data["EMA20"].shift(compare_period) - test_data["EMA20"])/test_data["EMA20"]) * 100 < -(signal_trigger),
        -1,
        test_data["Signal"],
    )

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

    # Split data into train and test sets

    X = train_data[IC.input_to_model]
    y = train_data["Signal"]
    X_test = test_data[IC.input_to_model]
    y_test = test_data["Signal"]
    #X_train, X_test, y_train, y_test = train_test_split(
    #    Xi, yi, test_size=0.2, random_state=42
    #)

    # Train logistic regression model
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(X, y)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(classification_report(y_test, y_pred))
    print(f"Increase Model Accuracy: {accuracy:.2f}")

    test_data['Singal_predict'] = y_pred
    test_data.to_csv(sample_path, sep=",")

    # decrease model
    #Xd = train_data_decrease[["EMA20","EMA50", "EMA200", "RSI"]]
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
    #    pickle.dump(model_i, file)

    with open(out_put_model, 'wb') as file:
        pickle.dump(model, file)
