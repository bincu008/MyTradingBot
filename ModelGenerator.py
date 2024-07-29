# Required libraries
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
bias_counter = 3
signal_trigger = 0.15 # percentage of price change
compare_period = -7
sample_increase_path = "C:\\Users\\Hieu\\OneDrive\\Desktop\\manipulate_increase.csv"
sample_decrease_path = "C:\\Users\\Hieu\\OneDrive\\Desktop\\manipulate_decrease.csv"

output_sell_model = "my_trained_model_sell.pkl"
output_buy_model = "my_trained_model_buy.pkl"

def GenerateModel():
    # Download historical data for XAU/USD (gold)
    gold_data = yf.download("GC=F", start="2024-06-15", end="2024-07-28", interval="5m")

    # Calculate 15-minute EMA
    gold_data["EMA200"] = gold_data["Close"].ewm(span=200).mean()
    gold_data["EMA20"] = gold_data["Close"].ewm(span=20).mean()
    gold_data["EMA50"] = gold_data["Close"].ewm(span=50).mean()

    # Calculate RSI
    delta = gold_data["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=20).mean()
    avg_loss = pd.Series(loss).rolling(window=20).mean()
    rs = avg_gain / avg_loss
    holder = 100 - (100 / (1 + rs))

    gold_data["RSI"] = pd.Series(dtype="double")
    if len(holder) == len(gold_data["RSI"]):
        for i in range(len(gold_data["RSI"])):
            gold_data["RSI"].values[i] = holder.values[i]

    gold_data = gold_data.iloc[200:, :]
    gold_data["Signal"] = pd.Series(dtype="double")

    # Create binary labels (1 for buy, 0 for sell) EMA cross below
    buy_signal = np.where(((gold_data["EMA20"].shift(compare_period) - gold_data["EMA20"])/gold_data["EMA20"]) * 100 > signal_trigger,
        1,
        0,
    )

    # Create binary labels (2 for sell, 0 for sell) EMA cross above
    sell_signal = np.where(((gold_data["EMA20"].shift(compare_period) - gold_data["EMA20"])/gold_data["EMA20"]) * 100 < -(signal_trigger),
        2,
        0,
    )
    merge_signal = pd.Series(dtype="double")

    # manipulating data
    if (len(buy_signal) == len(sell_signal)) and (
        len(buy_signal) == len(gold_data["Signal"])
    ):
        for i in range(len(gold_data["Signal"])):
            if (buy_signal[i] == 1) and (sell_signal[i] == 0):
                gold_data["Signal"].values[i] = 1
                continue
            gold_data["Signal"].values[i] = 0
            
        gold_data.to_csv("C:\\Users\\Hieu\\OneDrive\\Desktop\\manipulate_increase.csv", sep=",")
        gold_data["Signal"] = pd.Series(dtype="double")

        for i in range(len(gold_data["Signal"])):
            if (sell_signal[i] == 2) and (buy_signal[i] == 0):
                gold_data["Signal"].values[i] = 1
                continue
            gold_data["Signal"].values[i] = 0
            
        gold_data.to_csv("C:\\Users\\Hieu\\OneDrive\\Desktop\\manipulate_decrease.csv", sep=",")

    gold_data_increase = pd.read_csv(sample_increase_path)
    gold_data_decrease = pd.read_csv(sample_decrease_path)
    # Split data into train and test sets

    Xi = gold_data_increase[["EMA20","EMA50", "EMA200", "RSI"]]
    yi = gold_data_increase["Signal"]
    Xi_train, Xi_test, yi_train, yi_test = train_test_split(
        Xi, yi, test_size=0.2, random_state=42
    )

    # Train logistic regression model
    model_i = LogisticRegression()
    model_i.fit(Xi_train, yi_train)

    # Evaluate model
    yi_pred = model_i.predict(Xi_test)
    accuracy_i = (model_i.predict(Xi_test) == yi_test).mean()
    print(classification_report(yi_test, yi_pred))
    print(f"Increase Model Accuracy: {accuracy_i:.2f}")



    # decrease model
    Xd = gold_data_decrease[["EMA20","EMA50", "EMA200", "RSI"]]
    yd = gold_data_decrease["Signal"]
    Xd_train, Xd_test, yd_train, yd_test = train_test_split(
        Xd, yd, test_size=0.2, random_state=42
    )
    
    model_d = LogisticRegression()
    model_d.fit(Xd_train, yd_train)
    yd_pred = model_d.predict(Xd_test)
    accuracy_d = (model_d.predict(Xd_test) == yd_test).mean()
    print(classification_report(yd_test, yd_pred))
    print(f"Decrease Model Accuracy: {accuracy_d:.2f}")

    with open(output_buy_model, 'wb') as file:
        pickle.dump(model_i, file)

    with open(output_sell_model, 'wb') as file:
        pickle.dump(model_d, file)
