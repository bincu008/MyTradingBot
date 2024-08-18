import string
import pandas as pd
import numpy as np

class IndicatorTable:
    def __init__(self):
        pd.options.mode.chained_assignment = None  # default='warn'
        self.curtain = 20
        self.roll_back = 5
        self.signal_trigger = 0.1 # percentage of price change
        self.quick_trigger = 0.2
        self.compare_period_long = -15
        self.compare_period_short = -30
        self.regression_sensitivity = 0.0
        self.key_token = "none"
        self.input_to_model = ["RSI","MACD","ATR","tick_volume",#"DCxEMA20","DCxEMA50","DCxEMA200",
                      "dfEMA20","dfEMA50","dfEMA100","dfEMA200",
                      #"DEMA20x50","DEMA20x200","DEMA50x200",
                      #"CU_EMA20x50","CU_EMA20x200","CU_EMA50x200",
                      #"CL_EMA20x50","CL_EMA20x200","CL_EMA50x200",
                      "UpperxClose", "LowerxClose"]#"SMA","Upper","Lower",
                      #"dfEMA20x1","dfEMA50x1","dfEMA100x1","dfEMA200x1"]

                      #  ["close","RSI","MACD",#"DCxEMA20","DCxEMA50","DCxEMA200",
                      #"DEMA20x50","DEMA20x200","DEMA50x200",
                      ##"CU_EMA20x50","CU_EMA20x200","CU_EMA50x200",
                      ##"CL_EMA20x50","CL_EMA20x200","CL_EMA50x200",
                      #"dfEMA20x1","dfEMA50x1","dfEMA100x1","dfEMA200x1",
                      #"dfEMA20x2","dfEMA50x2","dfEMA100x2","dfEMA200x2"]
                      #"DiffCO","DiffCH","DiffCL","DiffOC"] 
                      #,"DCxEMA10""DEMA10x20","DEMA10x50","DEMA10x200","CU_EMA10x20","CU_EMA10x50","CU_EMA10x200",
                      # "CL_EMA10x20","CL_EMA10x50","CL_EMA10x200","dfEMA10","dfEMA20","dfEMA50",
    def Calculate(self, table):
        self.table = table
        if self.key_token == "HA":
            self.table['HA_Close'] = (table['open'] + table['high'] + table['low'] + table['close']) / 4
            self.table['HA_Open'] = table['HA_Close']
            self.table['HA_Open'] = (table['HA_Close'].shift(1) + table['HA_Open'].shift(1)) / 2
            self.table['HA_High'] = table[['high', 'HA_Open', 'HA_Close']].max(axis=1)
            self.table['HA_Low'] = table[['low', 'HA_Open', 'HA_Close']].min(axis=1)
            key_close = 'HA_Close'
            key_high = 'HA_High'
            key_low = 'HA_Low'
            #key_open = 'HA_Open'
        else:
            key_close = 'close'
            key_high = 'high'
            key_low = 'low'
            #key_open = 'open'

        # Calculate 15-minute EMA
        self.table["EMA200"] = self.table[key_close].ewm(span=200).mean()
        self.table["EMA100"] = self.table[key_close].ewm(span=100).mean()
        #table["EMA10"] = table[key_close].ewm(span=10).mean()
        self.table["EMA20"] = self.table[key_close].ewm(span=20).mean()
        self.table["EMA50"] = self.table[key_close].ewm(span=50).mean()

        # Calculate Bollinger Bands
        self.table['SMA'] = self.table[key_close].rolling(window=self.curtain).mean()
        self.table['Upper'] = self.table['SMA'] + 2 * self.table[key_close].rolling(window=self.curtain).std()
        self.table['Lower'] = self.table['SMA'] - 2 * self.table[key_close].rolling(window=self.curtain).std()
        self.table['UpperxClose'] = self.table[key_close] - table['Upper']
        self.table['LowerxClose'] = self.table[key_close] - table['Lower']

        # Calculate RSI
        delta = self.table[key_close].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=self.curtain).mean()
        avg_loss = pd.Series(loss).rolling(window=self.curtain).mean()
        rs = avg_gain / avg_loss
        holder = 100 - (100 / (1 + rs))

        self.table["RSI"] = pd.Series(dtype="double")
        if len(holder) == len(self.table["RSI"]):
            for i in range(len(self.table["RSI"])):
                self.table["RSI"].values[i] = holder.values[i]

        # calculate Average True Range
        tr = np.maximum(self.table[key_high] - self.table[key_low],
                        np.abs(self.table[key_high] - self.table[key_close].shift()))
        self.table['ATR'] = tr.rolling(window=self.curtain).mean()

        # calulate MACD
        self.table['EMA12'] = self.table[key_close].ewm(span=12).mean()
        self.table['EMA26'] = self.table[key_close].ewm(span=26).mean()
        self.table['MACD'] = self.table['EMA12'] - self.table['EMA26']

        # calulateStochastic Oscillator
        #table['LowestLow'] = table[key_low].rolling(window=curtain).min()
        #table['HighestHigh'] = table[key_high].rolling(window=curtain).max()
        #table['Stochastic'] = 100 * (table[key_close] - table['LowestLow']) / (table['HighestHigh'] - table['LowestLow'])


        # additional calculation
            # upper cut
        #table['CU_EMA10x20'] = np.where(((table['EMA10'].shift(1) > table['EMA20'].shift(1)) & (table['EMA10'] < table['EMA20'])),1 ,0)
        #table['CU_EMA10x50'] = np.where(((table['EMA10'].shift(1) > table['EMA50'].shift(1)) & (table['EMA10'] < table['EMA50'])),1 ,0)
        #table['CU_EMA10x200'] = np.where(((table['EMA10'].shift(1) > table['EMA200'].shift(1)) & (table['EMA10'] < table['EMA200'])),1 ,0)
        #table['CU_EMA20x50'] = np.where(((table['EMA20'].shift(1) > table['EMA50'].shift(1)) & (table['EMA20'] < table['EMA50'])),1 ,0)
        #table['CU_EMA20x200'] = np.where(((table['EMA20'].shift(1) > table['EMA200'].shift(1)) & (table['EMA20'] < table['EMA200'])),1 ,0)
        #table['CU_EMA50x200'] = np.where(((table['EMA50'].shift(1) > table['EMA200'].shift(1)) & (table['EMA50'] < table['EMA200'])),1 ,0)
            # lower cut
        #table['CL_EMA10x20'] = np.where(((table['EMA10'].shift(1) < table['EMA20'].shift(1)) & (table['EMA10'] > table['EMA20'])),1 ,0)
        #table['CL_EMA10x50'] = np.where(((table['EMA10'].shift(1) < table['EMA50'].shift(1)) & (table['EMA10'] > table['EMA50'])),1 ,0)
        #table['CL_EMA10x200'] = np.where(((table['EMA10'].shift(1) < table['EMA200'].shift(1)) & (table['EMA10'] > table['EMA200'])),1 ,0)
        #table['CL_EMA20x50'] = np.where(((table['EMA20'].shift(1) < table['EMA50'].shift(1)) & (table['EMA20'] > table['EMA50'])),1 ,0)
        #table['CL_EMA20x200'] = np.where(((table['EMA20'].shift(1) < table['EMA200'].shift(1)) & (table['EMA20'] > table['EMA200'])),1 ,0)
        #table['CL_EMA50x200'] = np.where(((table['EMA50'].shift(1) < table['EMA200'].shift(1)) & (table['EMA50'] > table['EMA200'])),1 ,0)
            # distance from what to what
        #table['DCxEMA10'] = (table[key_close] - table['EMA10']) / table[key_close]
        #table['DCxEMA20'] = (table[key_close] - table['EMA20'])
        #table['DCxEMA50'] = (table[key_close] - table['EMA50'])
        #table['DCxEMA200'] = (table[key_close] - table['EMA200'])

        #table['DEMA10x20'] = (table['EMA10'] - table['EMA20']) / table['EMA10']
        #table['DEMA10x50'] = (table['EMA10'] - table['EMA50']) / table['EMA10']
        #table['DEMA10x200'] = (table['EMA10'] - table['EMA200']) / table['EMA10']
        self.table['DEMA20x50'] = (self.table['EMA20'] - self.table['EMA50'])
        self.table['DEMA20x200'] = (self.table['EMA20'] - self.table['EMA200'])
        self.table['DEMA50x200'] = (self.table['EMA50'] - self.table['EMA200'])
            # open close high low difference
        #table['DiffCO'] = (table[key_close] - table[key_open]) / table[key_close]
        #table['DiffCH'] = (table[key_close] - table[key_high]) / table[key_close]
        #table['DiffCL'] = (table[key_close] - table[key_low]) / table[key_close]
        #table['DiffOC'] = (table[key_close] - table[key_high]) / table[key_close]

            # EMA change
        #table['dfEMA10'] = (table['EMA10'] - table['EMA10'].shift(roll_back)) / table['EMA10']
        self.table['dfEMA20'] = (self.table['EMA20'] - self.table['EMA20'].shift(5))
        self.table['dfEMA50'] = (self.table['EMA50'] - self.table['EMA50'].shift(5))
        self.table['dfEMA100'] = (self.table['EMA100'] - self.table['EMA100'].shift(5))
        self.table['dfEMA200'] = (self.table['EMA200'] - self.table['EMA200'].shift(5))

        #table['dfEMA20x2'] = (table['EMA20'].shift(roll_back) - table['EMA20'].shift(2*roll_back)) / table['EMA20'].shift(roll_back)
        #table['dfEMA50x2'] = (table['EMA50'].shift(roll_back) - table['EMA50'].shift(2*roll_back)) / table['EMA50'].shift(roll_back)
        #table['dfEMA100x2'] = (table['EMA100'].shift(roll_back) - table['EMA100'].shift(2*roll_back)) / table['EMA100'].shift(roll_back)
        #table['dfEMA200x2'] = (table['EMA200'].shift(roll_back) - table['EMA200'].shift(2*roll_back)) / table['EMA200'].shift(roll_back)

        # adding backward data

        for i in range(1, self.roll_back + 1):
            ratio = 3
            key = '_RB_'
            rsi_name = 'RSI' + key + str(i)
            macd_name = 'MACD' + key + str(i)
            atr_name = 'ATR' + key + str(i)
            bgbu_name = "UpperxClose" + key + str(i)
            bgbl_name = "LowerxClose" + key + str(i)
            #dema25_name = "DEMA20x50" + key + str(i)
            #dema22_name = "DEMA20x200" + key + str(i)
            #dema52_name = "DEMA50x200" + key + str(i)
            dema20_name = "dfEMA20" + key + str(i)
            dema50_name = "dfEMA50" + key + str(i)
            dema100_name = "dfEMA100" + key + str(i)
            dema200_name = "dfEMA200" + key + str(i)

            self.table[rsi_name] = self.table['RSI'].shift(i)
            self.table[macd_name] = self.table['MACD'].shift(i)
            self.table[atr_name] = self.table['ATR'].shift((i- 1)*ratio) - self.table['ATR'].shift(i)
            self.table[bgbu_name] = self.table['UpperxClose'].shift(i*ratio)
            self.table[bgbl_name] = self.table['LowerxClose'].shift(i*ratio)
            #table[dema25_name] = table['DEMA20x50'].shift(i*ratio)
            #table[dema22_name] = table['DEMA20x200'].shift(i*ratio)
            #table[dema52_name] = table['DEMA50x200'].shift(i*ratio)
            self.table[dema20_name] = self.table['dfEMA20'].shift(i*ratio)
            self.table[dema50_name] = self.table['dfEMA50'].shift(i*ratio)
            self.table[dema100_name] = self.table['dfEMA100'].shift(i*ratio)
            self.table[dema200_name] = self.table['dfEMA200'].shift(i*ratio)

            self.input_to_model.append(rsi_name)
            self.input_to_model.append(macd_name)
            self.input_to_model.append(atr_name)
            self.input_to_model.append(bgbu_name)
            self.input_to_model.append(bgbl_name)
            self.input_to_model.append(dema20_name)
            self.input_to_model.append(dema50_name)
            self.input_to_model.append(dema100_name)
            self.input_to_model.append(dema200_name)
            self.input_to_model = list(set(self.input_to_model))

        # remove first 200 rows, unused
        self.table = table.iloc[200:, :]
        #return table
    
    def ExportData(self):
        return self.table[self.input_to_model]

    def UpdatePrediction(self, y_pred, y_pred_proba):
        # manual offset
        for i in range(len(y_pred)):
            buy_prob = y_pred_proba[i][0] + self.regression_sensitivity
            neutral_prob = y_pred_proba[i][1] - 2 * self.regression_sensitivity
            sell_prob = y_pred_proba[i][2] + self.regression_sensitivity

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

        self.table.loc[:, 'Predict'] = y_pred
        self.table.loc[:, 'Predict_buy'] = y_pred_proba[:, 2]
        self.table.loc[:, 'Predict_neut'] = y_pred_proba[:, 1]
        self.table.loc[:, 'Predict_sell'] = y_pred_proba[:, 0]

    def DataManipulate(self):
        #table["Signal"] = np.where((((table["EMA20"].shift(compare_period_long) - table["EMA20"])/table["EMA20"]) * 100 > signal_trigger),
        #    2,
        #    0,
        #)

        #table["Signal"] = np.where(((((table["EMA20"].shift(compare_period_short) - table["EMA20"])/table["EMA20"]) * 100 > quick_trigger)
        #                                 & (table['dfEMA20x1'] > table['dfEMA20x2'])),
        #    1,
        #    table["Signal"],
        #)

        #table["Signal"] = np.where((((table["EMA20"].shift(compare_period_long) - table["EMA20"])/table["EMA20"]) * 100 < -(signal_trigger)),
        #    -2,
        #    table["Signal"],
        #)

        #table["Signal"] = np.where(((((table["EMA20"].shift(compare_period_short) - table["EMA20"])/table["EMA20"]) * 100 < -(quick_trigger))
        #                                 & (table['dfEMA20x1'] > table['dfEMA20x2'])),
        #    -1,
        #    table["Signal"],
        #)
        signal = pd.Series(dtype="int")
        signal = np.where((((self.table["EMA20"].shift(self.compare_period_long) - self.table["EMA20"])/self.table["EMA20"]) * 100 > self.signal_trigger),
            1,
            0,
        )

        signal = np.where((((self.table["EMA20"].shift(self.compare_period_long) - self.table["EMA20"])/self.table["EMA20"]) * 100 < -(self.signal_trigger)),
            -1,
            signal,
        )

        self.table.loc[:, 'Signal'] = signal
        return signal