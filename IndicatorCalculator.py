import pandas as pd
import numpy as np

curtain = 14
roll_back = 7
input_to_model = ["RSI","MACD",#"DCxEMA20","DCxEMA50","DCxEMA200",
                  "DEMA20x50","DEMA20x200","DEMA50x200",
                  #"CU_EMA20x50","CU_EMA20x200","CU_EMA50x200",
                  #"CL_EMA20x50","CL_EMA20x200","CL_EMA50x200",
                  "dfEMA150","dfEMA200"]
                  #"DiffCO","DiffCH","DiffCL","DiffOC"] ,"DCxEMA10""DEMA10x20","DEMA10x50","DEMA10x200","CU_EMA10x20","CU_EMA10x50","CU_EMA10x200",
                  # "CL_EMA10x20","CL_EMA10x50","CL_EMA10x200","dfEMA10","dfEMA20","dfEMA50",
def IndicatorCalculator(table, key_token):
    if key_token == "Test":
        key_close = 'Close'
        key_high = 'High'
        key_low = 'Low'
        key_open = 'Open'
    else:
        key_close = 'close'
        key_high = 'high'
        key_low = 'low'
        key_open = 'open'

    # Calculate 15-minute EMA
    table["EMA200"] = table[key_close].ewm(span=200).mean()
    table["EMA150"] = table[key_close].ewm(span=150).mean()
    #table["EMA10"] = table[key_close].ewm(span=10).mean()
    table["EMA20"] = table[key_close].ewm(span=20).mean()
    table["EMA50"] = table[key_close].ewm(span=50).mean()

    # Calculate RSI
    delta = table[key_close].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=curtain).mean()
    avg_loss = pd.Series(loss).rolling(window=curtain).mean()
    rs = avg_gain / avg_loss
    holder = 100 - (100 / (1 + rs))

    table["RSI"] = pd.Series(dtype="double")
    if len(holder) == len(table["RSI"]):
        for i in range(len(table["RSI"])):
            table["RSI"].values[i] = holder.values[i]

    # calculate Average True Range
    tr = np.maximum(table[key_high] - table[key_low],
                    np.abs(table[key_high] - table[key_close].shift()))
    table['ATR'] = tr.rolling(window=curtain).mean()

    # calulate MACD
    table['EMA12'] = table[key_close].ewm(span=12).mean()
    table['EMA26'] = table[key_close].ewm(span=26).mean()
    table['MACD'] = table['EMA12'] - table['EMA26']

    # calulateStochastic Oscillator
    table['LowestLow'] = table[key_low].rolling(window=curtain).min()
    table['HighestHigh'] = table[key_high].rolling(window=curtain).max()
    table['Stochastic'] = 100 * (table[key_close] - table['LowestLow']) / (table['HighestHigh'] - table['LowestLow'])


    # additional calculation
        # upper cut
    #table['CU_EMA10x20'] = np.where(((table['EMA10'].shift(1) > table['EMA20'].shift(1)) & (table['EMA10'] < table['EMA20'])),1 ,0)
    #table['CU_EMA10x50'] = np.where(((table['EMA10'].shift(1) > table['EMA50'].shift(1)) & (table['EMA10'] < table['EMA50'])),1 ,0)
    #table['CU_EMA10x200'] = np.where(((table['EMA10'].shift(1) > table['EMA200'].shift(1)) & (table['EMA10'] < table['EMA200'])),1 ,0)
    table['CU_EMA20x50'] = np.where(((table['EMA20'].shift(1) > table['EMA50'].shift(1)) & (table['EMA20'] < table['EMA50'])),1 ,0)
    table['CU_EMA20x200'] = np.where(((table['EMA20'].shift(1) > table['EMA200'].shift(1)) & (table['EMA20'] < table['EMA200'])),1 ,0)
    table['CU_EMA50x200'] = np.where(((table['EMA50'].shift(1) > table['EMA200'].shift(1)) & (table['EMA50'] < table['EMA200'])),1 ,0)
        # lower cut
    #table['CL_EMA10x20'] = np.where(((table['EMA10'].shift(1) < table['EMA20'].shift(1)) & (table['EMA10'] > table['EMA20'])),1 ,0)
    #table['CL_EMA10x50'] = np.where(((table['EMA10'].shift(1) < table['EMA50'].shift(1)) & (table['EMA10'] > table['EMA50'])),1 ,0)
    #table['CL_EMA10x200'] = np.where(((table['EMA10'].shift(1) < table['EMA200'].shift(1)) & (table['EMA10'] > table['EMA200'])),1 ,0)
    table['CL_EMA20x50'] = np.where(((table['EMA20'].shift(1) < table['EMA50'].shift(1)) & (table['EMA20'] > table['EMA50'])),1 ,0)
    table['CL_EMA20x200'] = np.where(((table['EMA20'].shift(1) < table['EMA200'].shift(1)) & (table['EMA20'] > table['EMA200'])),1 ,0)
    table['CL_EMA50x200'] = np.where(((table['EMA50'].shift(1) < table['EMA200'].shift(1)) & (table['EMA50'] > table['EMA200'])),1 ,0)
        # distance from what to what
    #table['DCxEMA10'] = (table[key_close] - table['EMA10']) / table[key_close]
    table['DCxEMA20'] = (table[key_close] - table['EMA20']) / table[key_close]
    table['DCxEMA50'] = (table[key_close] - table['EMA50']) / table[key_close]
    table['DCxEMA200'] = (table[key_close] - table['EMA200']) / table[key_close]

    #table['DEMA10x20'] = (table['EMA10'] - table['EMA20']) / table['EMA10']
    #table['DEMA10x50'] = (table['EMA10'] - table['EMA50']) / table['EMA10']
    #table['DEMA10x200'] = (table['EMA10'] - table['EMA200']) / table['EMA10']
    table['DEMA20x50'] = (table['EMA20'] - table['EMA50']) / table['EMA20']
    table['DEMA20x200'] = (table['EMA20'] - table['EMA200']) / table['EMA20']
    table['DEMA50x200'] = (table['EMA50'] - table['EMA200']) / table['EMA50']
        # open close high low difference
    table['DiffCO'] = (table[key_close] - table[key_open]) / table[key_close]
    table['DiffCH'] = (table[key_close] - table[key_high]) / table[key_close]
    table['DiffCL'] = (table[key_close] - table[key_low]) / table[key_close]
    table['DiffOC'] = (table[key_close] - table[key_high]) / table[key_close]

        # EMA change
    #table['dfEMA10'] = (table['EMA10'] - table['EMA10'].shift(roll_back)) / table['EMA10']
    table['dfEMA20'] = (table['EMA20'] - table['EMA20'].shift(roll_back)) / table['EMA20']
    table['dfEMA50'] = (table['EMA50'] - table['EMA50'].shift(roll_back)) / table['EMA50']
    table['dfEMA150'] = (table['EMA150'] - table['EMA150'].shift(roll_back)) / table['EMA150']
    table['dfEMA200'] = (table['EMA200'] - table['EMA200'].shift(roll_back)) / table['EMA200']

    # remove first 200 rows, unused
    table = table.iloc[200:, :]
    return table
