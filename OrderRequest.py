import MetaTrader5 as MT5

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

def validate_buy(pred):
    if ((pred[-1] == 1) and (pred[-10:].count(1) < 4)):
        return True
    return False

def validate_sell(pred):
    if ((pred[-1] == -1) and (pred[-10:].count(-1) < 4)):
        return True
    return False

def create_send_order(option, offset):
    infor = MT5.symbol_info_tick(trading_symbol)
    buy_price = infor.ask
    sell_price = infor.bid

    if (option == "buy"):
        request_buy["price"] = buy_price
        request_buy["sl"] = buy_price - (1*offset) # 2 dollar please
        request_buy["tp"] = buy_price + (2*offset)
        result = MT5.order_send(request_buy)
        txt = f"Order status: {result}"
        print(txt)
        return txt

    if (option == "sell"):
        request_sell["price"] = sell_price
        request_sell["sl"] = sell_price + (1*offset) # 2 dollar please
        request_sell["tp"] = sell_price - (2*offset)
        result = MT5.order_send(request_sell)
        txt = f"Order status: {result}"
        print(txt)
        return txt
