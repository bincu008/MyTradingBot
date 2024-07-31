# Required libraries
import MetaTrader5 as MT5
import pandas as pd
import numpy as np
import IndicatorCalculator as IC
from datetime import datetime
from datetime import timedelta
import pytz
import os
import pickle
import time
import ModelGenerator as MG

generate_model = True
ID = 312299110
PW = 'Mituongden123!'
SV = 'XMGlobal-MT5 7'

if __name__ == "__main__":
    if MT5.initialize():
        MT5.login(ID, PW, SV)
        account_info = MT5.account_info()
        terminal_info=MT5.terminal_info()
        print(account_info)
        print(terminal_info)
    else:
        print("initialize() failed")
        MT5.shutdown()
        
    utc_time = pytz.timezone('UTC')
    # stupid server is UTC + 3!
    noww = datetime.now(utc_time) + timedelta(hours=3)
    date_ = noww - timedelta(days =30)
    date_from = date_.replace() #(hour=0, minute=0, second=0, microsecond=0)
    date_to = noww
    
    date_from_train = noww - timedelta(days = 90)
    date_to_train = noww - timedelta(days = 30)

    if (generate_model):
        train_data = pd.DataFrame(MT5.copy_rates_range("GOLD#", MT5.TIMEFRAME_M3, date_from_train, date_to_train))
        MG.GenerateModel(train_data)
    elif (not os.path.isfile(MG.output_sell_model) or not os.path.isfile(MG.output_buy_model)):
        train_data = pd.DataFrame(MT5.copy_rates_range("GOLD#", MT5.TIMEFRAME_M3, date_from_train, date_to_train))
        MG.GenerateModel(train_data)
    
    with open(MG.out_put_model, 'rb') as file:
        my_model = pickle.load(file)
        

    while True:
        # connect to MetaTrader 5
        if not MT5.initialize():
            print("initialize() failed")
            MT5.shutdown()

        # request 1000 ticks from EURAUD
        now = datetime.now()
        gold_ticks = pd.DataFrame(MT5.copy_rates_range("GOLD#", MT5.TIMEFRAME_M3, date_from, date_to))
        gold_ticks = IC.IndicatorCalculator(gold_ticks, "data")
        gold_ticks.to_csv(mg.real_data_path)

        Test = gold_ticks[IC.input_to_model]
        pred = my_model.predict(Test)
        
        #price = np.average([gold_ticks[0]['ask'], gold_ticks[0]['bid']])
        print(f"time : {now} current price ask: {gold_ticks[0]['ask']:.2f} current price bid:{gold_ticks[0]['bid']:.2f}")
        time.sleep(10)
