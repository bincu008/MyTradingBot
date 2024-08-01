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

generate_model = False
one_minute_model = "my_trained_model_1m.pkl"
log_file = "log.txt"

ID = 312299110
PW = 'Mituongden123!'
SV = 'XMGlobal-MT5 7'
trading_symbol = "GOLD#"
lot = 0.01
price = 0.0
spread = 0.0

request_buy = {
    "action": MT5.TRADE_ACTION_DEAL,
    "symbol": trading_symbol,
    "volume": lot,
    "type": MT5.ORDER_TYPE_BUY,
    "price": price,
    "sl": price - 100,
    "tp": price + 100,
    "comment": "python script open",
    "type_time": MT5.ORDER_TIME_GTC,
    "type_filling": MT5.ORDER_FILLING_IOC,
}

request_sell = {
    "action": MT5.TRADE_ACTION_DEAL,
    "symbol": trading_symbol,
    "volume": lot,
    "type": MT5.ORDER_TYPE_SELL,
    "price": price,
    "sl": price - 100,
    "tp": price + 100,
    "comment": "python script open",
    "type_time": MT5.ORDER_TIME_GTC,
    "type_filling": MT5.ORDER_FILLING_IOC,
}


if __name__ == "__main__":
    if MT5.initialize():
        MT5.login(ID, PW, SV)
        account_info = MT5.account_info()
        spread = MT5.symbol_info(trading_symbol).spread
        print(account_info)
        print(MT5.terminal_info())
        print(f"\n\n===== TRADING SYMBOL [{trading_symbol}] spread =[{spread}] =====")
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
        train_data = pd.DataFrame(MT5.copy_rates_range(trading_symbol, MT5.TIMEFRAME_M1, date_from_train, date_to_train))
        test_data = pd.DataFrame(MT5.copy_rates_range(trading_symbol, MT5.TIMEFRAME_M1, date_to_train, date_to))
        MG.GenerateModel(train_data, test_data)
    
    with open(one_minute_model, 'rb') as file:
        my_model = pickle.load(file)
        

    while True:
        log_list = []
        # connect to MetaTrader 5
        if not MT5.initialize():
            print("initialize() failed")
            MT5.shutdown()

        spread = MT5.symbol_info(trading_symbol).spread
        # if spread < 1.0:
        now = datetime.now(utc_time) + timedelta(hours=3)
        date_from = (noww - timedelta(days =30)).replace(hour=0, minute=0, second=0, microsecond=0)

        gold_ticks = pd.DataFrame(MT5.copy_rates_range(trading_symbol, MT5.TIMEFRAME_M1, date_from, now))
        gold_ticks = IC.IndicatorCalculator(gold_ticks, "Real")
            
        buy_price = MT5.symbol_info_tick(trading_symbol).ask
        sell_price = MT5.symbol_info_tick(trading_symbol).bid

        Test = gold_ticks[IC.input_to_model]
        pred = my_model.predict(Test)

        my_pos = MT5.positions_get()
        history_order = MT5.history_orders_get(date_from,now)

        if (("tp" not in history_order[-1].comment) & ("tp" not in history_order[-2].comment)):
            flag = True
        else:
            flag = False
        txt = f"time: {now} current price ask: {buy_price} current price bid:{sell_price} prediction: {pred[-1]} trade block: {flag}"
        log_list.append(txt)
        #print(txt)
        
        if ((len(my_pos) == 0) and (flag == False)):
            if ((pred[-1] == 1) and (pred[-10:].count(1) < 4)): # buy predict, not too late to enter
                request_buy["price"] = buy_price
                request_buy["sl"] = buy_price - (1*gold_ticks.iloc[-1]['ATR']) # 2 dollar please
                request_buy["tp"] = buy_price + (1.5*gold_ticks.iloc[-1]['ATR'])
                result = MT5.order_send(request_buy)
                txt = f"Order status: {result}"
                log_list.append(txt)
                print(txt)
                time.sleep(600)

            elif ((pred[-1] == -1) and (pred[-10:].count(-1) < 4)): # sell predict, not too late to enter
                request_sell["price"] = sell_price
                request_sell["sl"] = sell_price + (1*gold_ticks.iloc[-1]['ATR']) # 2 dollar please
                request_sell["tp"] = sell_price - (1.5*gold_ticks.iloc[-1]['ATR'])
                result = MT5.order_send(request_sell)
                txt = f"Order status: {result}"
                log_list.append(txt)
                print(txt)
                time.sleep(600)
        else:
            txt = f"Position avalable, skip"
            log_list.append(txt)
            print(txt)
        
        file = open(log_file, "a")  # append mode
        for line in log_list:
            file.write(f"{line}\n")
        file.close()
        
        time.sleep(50)
