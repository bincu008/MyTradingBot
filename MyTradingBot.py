# Required libraries
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
bias_counter = 3

def bias_evaluator():
    print("Hello from a function")
    if (1 == 1):
        return 1
    return 2

def buy_rule():
    print("Hello from a function")

def sell_rule():
    print("Hello from a function")

def cut_above(EMA20, EMA50):
    if ((EMA20.shift(1) > EMA50.shift(1)) and (EMA20 <= EMA50)):
        return True
    return False

def cut_below(EMA20, EMA50):
    if ((EMA20.shift(1) < EMA50.shift(1)) and (EMA20 >= EMA50)):
        return True
    return False

if __name__ == "__main__":
    # Download historical data for XAU/USD (gold)
    gold_data = yf.download("GC=F", start="2024-05-30", end="2024-07-28", interval="5m")

    # Calculate 15-minute EMA
    gold_data["EMA200"] = gold_data["Close"].ewm(span=200).mean()
    gold_data["EMA20"] = gold_data["Close"].ewm(span=20).mean()
    gold_data["EMA50"] = gold_data["Close"].ewm(span=50).mean()

    # Calculate RSI
    delta = gold_data["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    holder = 100 - (100 / (1 + rs))

    gold_data["RSI"] = pd.Series(dtype="double")
    if len(holder) == len(gold_data["RSI"]):
        for i in range(len(gold_data["RSI"])):
            gold_data["RSI"].values[i] = holder.values[i]

    gold_data = gold_data.iloc[200:, :]
    gold_data["Signal"] = pd.Series(dtype="double")

    # Create binary labels (1 for buy, 0 for sell) EMA cross below
    buy_signal = np.where(
        (gold_data["RSI"] > 50) & ((gold_data["EMA20"].shift(1) < gold_data["EMA50"].shift(1)) & (gold_data["EMA20"] > gold_data["EMA50"]) & 
                                   ((gold_data["Close"] > gold_data["EMA200"]) & (gold_data["Close"].shift(1) > gold_data["EMA200"]) & 
                                    (gold_data["Close"].shift(2) > gold_data["EMA200"]) & (gold_data["Close"].shift(3) > gold_data["EMA200"]) &
                                    (gold_data["Close"].shift(4) > gold_data["EMA200"]))),
        1,
        0,
    )

    # Create binary labels (2 for sell, 0 for sell) EMA cross above
    sell_signal = np.where(
        (gold_data["RSI"] < 50) & ((gold_data["EMA20"].shift(1) > gold_data["EMA50"].shift(1)) & (gold_data["EMA20"] < gold_data["EMA50"]) & 
                                   ((gold_data["Close"] < gold_data["EMA200"]) & (gold_data["Close"].shift(1) < gold_data["EMA200"]) & 
                                    (gold_data["Close"].shift(2) < gold_data["EMA200"]) & (gold_data["Close"].shift(3) < gold_data["EMA200"]) &
                                    (gold_data["Close"].shift(4) < gold_data["EMA200"]))),
        2,
        0,
    )
    merge_signal = pd.Series(dtype="double")

    if (len(buy_signal) == len(sell_signal)) and (
        len(buy_signal) == len(gold_data["Signal"])
    ):
        for i in range(len(gold_data["Signal"])):
            if (buy_signal[i] == 1) and (sell_signal[i] == 0):
                gold_data["Signal"].values[i] = 1
                continue

            if (sell_signal[i] == 2) and (buy_signal[i] == 0):
                gold_data["Signal"].values[i] = 2
                continue

            if (sell_signal[i] == 2) and (buy_signal[i] == 1):
                gold_data["Signal"].values[i] = -1
                continue

            gold_data["Signal"].values[i] = 0

    gold_data.to_csv("C:\\Users\\Hieu\\OneDrive\\Desktop\\result.csv", sep=",")

    # Split data into train and test sets
    X = gold_data[["EMA200", "RSI"]]
    y = gold_data["Signal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Now you can integrate this model with MetaTrader 5 for live trading!
