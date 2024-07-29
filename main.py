# Required libraries
import MetaTrader5 as mt5
import numpy as np
from datetime import datetime
from datetime import timedelta
import pytz
import os
import pickle
import time
import ModelGenerator as mg

generate_model = False
ID = 312299110
PW = 'Mituongden123!'
SV = 'XMGlobal-MT5 7'

if __name__ == "__main__":
    if (generate_model):
        mg.GenerateModel()
    elif (not os.path.isfile(mg.output_sell_model) or not os.path.isfile(mg.output_buy_model)):
        mg.GenerateModel()
    
    with open(mg.output_buy_model, 'rb') as file:
        buy_model = pickle.load(file)

    with open(mg.output_sell_model, 'rb') as file:
        sell_model = pickle.load(file)
    
    if mt5.initialize():
        mt5.login(ID, PW, SV)
        account_info = mt5.account_info()
        terminal_info=mt5.terminal_info()
        symbols=mt5.symbol_info("Apple")._asdict()
        print(account_info)
        print(terminal_info)
        print(symbols)
    else:
        print("initialize() failed")
        mt5.shutdown()

    #moscow_tz = pytz.timezone('Europe/Moscow')
    utc_time = pytz.timezone('UTC')
    # stupid server is UTC + 3!
    noww = datetime.now(utc_time) + timedelta(hours=3)
    date_ = noww - timedelta(days =2)
    date_from = date_.replace() #(hour=0, minute=0, second=0, microsecond=0)
    date_to = noww
    #ticks = mt5.copy_ticks_range(self.symbol, date_from, date_to, mt5.COPY_TICKS_ALL)
    

    while True:
        # connect to MetaTrader 5
        if not mt5.initialize():
            print("initialize() failed")
            mt5.shutdown()

        # request 1000 ticks from EURAUD
        now = datetime.now()
        gold_ticks = mt5.copy_rates_range("GOLD#", mt5.TIMEFRAME_M5, date_from, date_to)
        gold_current_ticks = mt5.symbol_info_tick("GOLD#")
        #price = np.average([gold_ticks[0]['ask'], gold_ticks[0]['bid']])
        print(f"time : {now} current price ask: {gold_ticks[0]['ask']:.2f} current price bid:{gold_ticks[0]['bid']:.2f}")
        time.sleep(5)
