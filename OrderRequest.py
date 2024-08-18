import MetaTrader5 as MT5

class MT_trade_manager:
    def __init__(self):
        self.ID = 312299110
        self.PW = 'Mituongden123!'
        self.SV = 'XMGlobal-MT5 7'
        self.trading_symbol = "GOLD#"
        self.lot = 0.01
        self.price = 0.0
        self.spread = 0.0
        self.order_taken = []

        self.request_buy = {
        "action": MT5.TRADE_ACTION_DEAL,
        "symbol": self.trading_symbol,
        "volume": self.lot,
        "type": MT5.ORDER_TYPE_BUY,
        "price": self.price,
        "sl": self.price - 100,
        "tp": self.price + 100,
        "comment": "python script",
        "type_time": MT5.ORDER_TIME_GTC,
        "type_filling": MT5.ORDER_FILLING_IOC,
        }

        self.request_sell = {
        "action": MT5.TRADE_ACTION_DEAL,
        "symbol": self.trading_symbol,
        "volume": self.lot,
        "type": MT5.ORDER_TYPE_SELL,
        "price": self.price,
        "sl": self.price - 100,
        "tp": self.price + 100,
        "comment": "python script",
        "type_time": MT5.ORDER_TIME_GTC,
        "type_filling": MT5.ORDER_FILLING_IOC,
        }
    
    def login_account(self):
        MT5.login(self.ID, self.PW, self.SV)
        account_info = MT5.account_info()
        print(account_info)
        print(MT5.terminal_info())
        print(f"\n\n===== TRADING SYMBOL [{self.trading_symbol}] =====")

    def verify_order_status(self, my_pos, history_order):
        for i in range(len(self.order_taken)):
            if (self.order_taken[i]["Status"] == "Open" and self.order_taken[i]["Trade ID"].comment == 'Request executed'):
                for order in history_order:
                    if (order.ticket == self.order_taken[i]["Trade ID"].order and "sl" in order.comment):
                        self.order_taken[i]["Status"] == "Lose"
                    elif (order.ticket == self.order_taken[i]["Trade ID"].order and "tp" in order.comment):
                        self.order_taken[i]["Status"] == "Win"
        
        if (len(my_pos) > 0):
            return False
        if (len(self.order_taken) > 2):
            if (self.order_taken[-1]["Status"] == "Lose" and self.order_taken[-2]["Status"] == "Lose"):
                print("2 losing streak, need restart!!!!")
                return False
        return True

    def validate_buy(self, pred):
        if ((pred[-1] == 1) and (list(pred[-5:]).count(1) > 2)):
            return True
        return False

    def validate_sell(self, pred):
        if ((pred[-1] == -1) and (list(pred[-5:]).count(-1) > 2)):
            return True
        return False

    def check_for_trade(self, pred, offset):
        infor = MT5.symbol_info_tick(self.trading_symbol)
        buy_price = infor.ask
        sell_price = infor.bid
        self.spread = buy_price - sell_price
        if (self.spread > offset):
            guard_band = 2*self.spread
        else:
            guard_band = offset

        if (self.validate_buy(pred)):
            self.request_buy["price"] = buy_price
            self.request_buy["sl"] = buy_price - (1*guard_band) # 2 dollar please
            self.request_buy["tp"] = buy_price + (2*guard_band)
            result = MT5.order_send(self.request_buy)
            txt = f"Order status: {result}"
            if result.comment == 'Request executed':
                self.order_taken.append({"Trade ID" : result, "Status" : "Open"})
            print(txt)
            return {"result" : True, "message" : result}

        elif (self.validate_sell(pred)):
            self.request_sell["price"] = sell_price
            self.request_sell["sl"] = sell_price + (1*guard_band) # 2 dollar please
            self.request_sell["tp"] = sell_price - (2*guard_band)
            result = MT5.order_send(self.request_sell)
            txt = f"Order status: {result}"
            if result.comment == 'Request executed':
                self.order_taken.append({"Trade ID" : result, "Status" : "Open"})
            print(txt)
            return {"result" : True, "message" : result}
        return {"result" : False, "message" : ""}
    
    def trade_summary(self):
        win = 0
        lose = 0
        for order in self.order_taken:
            if order["Status"] == "Win":
                win += 1
            if order["Status"] == "Lose":
                lose += 1
        return {"win" : win, "lose" : lose}
