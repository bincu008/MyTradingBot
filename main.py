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
from sklearn.preprocessing import MinMaxScaler
import OrderRequest as OR
generate_model = True
one_minute_model = "my_trained_model_1m_normalized.pkl"
log_file = "log.txt"


if __name__ == "__main__":
    if MT5.initialize():
        MT5.login(OR.ID, OR.PW, OR.SV)
        account_info = MT5.account_info()
        print(account_info)
        print(MT5.terminal_info())
        print(f"\n\n===== TRADING SYMBOL [{OR.trading_symbol}] =====")
    else:
        print("initialize() failed")
        MT5.shutdown()
        
    utc_time = pytz.timezone('UTC')
    # stupid server is UTC + 3!
    noww = datetime.now(utc_time) + timedelta(hours=3)
    date_ = noww - timedelta(days =5)
    date_from = date_.replace() #(hour=0, minute=0, second=0, microsecond=0)
    date_to = noww - timedelta(days =1) 
    
    date_from_train = noww - timedelta(days = 62)
    date_to_train = noww - timedelta(days = 32)

    if (generate_model):
        train_data = pd.DataFrame(MT5.copy_rates_range(OR.trading_symbol, MT5.TIMEFRAME_M3, date_from_train, date_to_train))
        test_data = pd.DataFrame(MT5.copy_rates_range(OR.trading_symbol, MT5.TIMEFRAME_M3, date_to_train, date_to))
        MG.GenerateModel(train_data, test_data)
    
    with open(one_minute_model, 'rb') as file:
        my_model = pickle.load(file)
        

    while True:
        log_list = []
        try:
            if not MT5.initialize():
                print("initialize() failed")
                MT5.shutdown()
                
            # if spread < 1.0:
            now = datetime.now(utc_time) + timedelta(hours=3)
            date_from = (noww - timedelta(days =30)).replace(hour=0, minute=0, second=0, microsecond=0)

            gold_ticks = pd.DataFrame(MT5.copy_rates_range(OR.trading_symbol, MT5.TIMEFRAME_M1, date_from, now))
            gold_ticks = IC.IndicatorCalculator(gold_ticks, "none")
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(gold_ticks[IC.input_to_model])
            
        
            pred = my_model.predict(normalized_data)[-50:]

            my_pos = MT5.positions_get()
            history_order = MT5.history_orders_get(date_from,now)

            if (("tp" not in history_order[-1].comment) & ("tp" not in history_order[-2].comment)):
                flag = False
            else:
                flag = False
            txt = f"time: {now} current price ask: {MT5.symbol_info_tick(OR.trading_symbol).ask} current price bid:{MT5.symbol_info_tick(OR.trading_symbol).bid} prediction: {pred[-1]} trade block: {flag}"
            log_list.append(txt)
            print(txt)
        
            if ((len(my_pos) == 0) and (flag == False)):
            
                result = OR.create_send_order("sell", gold_ticks.iloc[-1]['ATR'])
                if (OR.validate_buy(pred)): # buy predict, not too late to enter
                    result = OR.create_send_order("buy", gold_ticks.iloc[-1]['ATR'])
                    log_list.append(txt)
                    time.sleep(300)

                elif (OR.validate_sell(pred)): # sell predict, not too late to enter
                    result = OR.create_send_order("sell", gold_ticks.iloc[-1]['ATR'])
                    log_list.append(txt)
                    time.sleep(300)
            else:
                txt = f"Position avalable, skip"
                log_list.append(txt)
                print(txt)
        except:
            txt = f"time: {now} error while executing code, sleep for 5m"
            file = open(log_file, "a")  # append mode
            print(txt)
            for line in log_list:
                file.write(f"{line}\n")
                file.close()
            time.sleep(300)
        
        file = open(log_file, "a")  # append mode
        for line in log_list:
            file.write(f"{line}\n")
        file.close()
        
        time.sleep(5)
