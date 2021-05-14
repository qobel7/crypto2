import ccxt
import pandas as pd
import numpy as np

from datetime import datetime

import plotly.graph_objects as plt
from plotly.subplots import make_subplots
from Indicators import Indicators
from csv import DictWriter
from datetime import datetime
from time import sleep
class Operation:
    indicatorClass=None
    lastSignal=None
    lastSignalT3=None
    def __init__(self):
        self.indicatorClass=Indicators()
        
    def placeOrder(self,exchange, pair, side, amount, price):
        # if side == 'buy':
        #     order = exchange.createLimitBuyOrder(pair, amount, price)
        # elif side == 'sell':
        #     order = exchange.createLimitSellOrder(pair, amount, price)
        if side == 'buy':
            order = exchange.create_order(pair,'market' ,'buy',amount*100)
        elif side == 'sell':
            order = exchange.create_order(pair,'market' ,'sell',amount)   
        print(order)
        return order

    def buy(self,exchange, pair, amount,conf):
        try:
            order_book = exchange.fetchOrderBook(pair)
            price = order_book['bids'][0][0]
            order = self.placeOrder(exchange, pair, 'buy', amount, price)
            amount = amount - order['filled']
            free = exchange.fetchOrder(order['id'])["price"]*exchange.fetchBalance()[conf['symbol']]['free'];
            free = free + exchange.fetchBalance()['USDT']['free']
            self.writeLogSelBuy("buy",  pair,exchange.fetchOrder(order['id'])["price"],conf,exchange,free)

            while True:
                order = exchange.fetchOrder(order['id'])
                if order['status'] == 'closed':
                    break
                amount = amount - order['filled']
                order_book = exchange.fetchOrderBook(pair)
                price = order_book['bids'][0][0]
                if price != order['price']:
                    print(f'price {price}, order {order["price"]}')
                    exchange.cancelOrder(order['id'])
                    order = self.placeOrder(exchange, pair, 'buy', amount, price)
        except:
            print("An exception occurred")
        

    def sell(self,exchange, pair, amount,conf):
        try:
            order_book = exchange.fetchOrderBook(pair)
            price = order_book['asks'][0][0]
            order = self.placeOrder(exchange, pair, 'sell', amount, price)
            amount = amount - order['filled']
            free = exchange.fetchOrder(order['id'])["price"]*exchange.fetchBalance()[conf['symbol']]['free'];
            free = free + exchange.fetchBalance()['USDT']['free']
            self.writeLogSelBuy("sell",  pair,exchange.fetchOrder(order['id'])["price"],conf,exchange,free)
            while True:
                order = exchange.fetchOrder(order['id'])
                if order['status'] == 'closed':
                    break
                amount = amount - order['filled']
                order_book = exchange.fetchOrderBook(pair)
                price = order_book['asks'][0][0]
                if price != order['price']:
                    exchange.cancelOrder(order['id'])
                    order = placeOrder(exchange, pair, 'sell', amount, price)
        except e:
            print("An exception occurred"+e)


    def cross(self,long, short):
        position = short > long
        pre_position = position.shift(1)
        result = np.where(position == pre_position, False, True)
        return result

    def cross_over(self,long, short):
        position = short > long
        pre_position = position.shift(1)
        result = np.logical_and(np.where(position == pre_position, False, True), np.equal(position, False))
        return result

    def cross_under(self,long, short):
        position = short > long
        pre_position = position.shift(1)
        result = np.logical_and(np.where(position == pre_position, False, True), np.equal(position, True))
        return result

    def supertrend_signal(self,df, super_trend_column):
        position = df[super_trend_column]
        pre_position = position.shift(1)
        conditions = [
            np.logical_and(np.where(position == pre_position, False, True), np.equal(position, 'up')) == True,
            np.logical_and(np.where(position == pre_position, False, True), np.equal(position, 'down')) == True
        ]
        choices = ['buy', 'sell']
        df['Signal'] = np.select(conditions, choices, default=np.NaN)

        return df

    def tilson_t3_signal(self,df, tilson_t3_column):
        position = df[tilson_t3_column]
        t3_last = position.shift(1)
        t3_previous = position.shift(2)
        t3_prev_previous = position.shift(3)
        
        conditions = [
            np.logical_and(np.where(t3_last > t3_previous, True, False), np.where(t3_previous < t3_prev_previous, True, False)) == True,
            np.logical_and(np.where(t3_last < t3_previous, True, False), np.where(t3_last > t3_previous, True, False)) == True
        ]
        choices = ['buy', 'sell']
        df['Signal'] = np.select(conditions, choices, default=np.NaN)

        return df

    def tilson_t3_strategy(self,df, tilson_t3_column):
        df = tilson_t3_signal(df, tilson_t3_column)
        signal = df['Signal']
        close = df['Close']
        df['buy_price'] = np.select([np.equal(signal, 'buy')], [close], default=np.nan)
        df['sell_price'] = np.select([np.equal(signal, 'sell')], [close], default=np.nan)
        return callculate_strategy_gain(df['Date'], signal, close)

    def supertrend_strategy(self,df, super_trend_column):
        df = self.supertrend_signal(df, super_trend_column)
        signal = df['Signal']
        close = df['Close']
        df['buy_price'] = np.select([np.equal(signal, 'buy')], [close], default=np.nan)
        df['sell_price'] = np.select([np.equal(signal, 'sell')], [close], default=np.nan)
        return self.callculate_strategy_gain(df['Date'], signal, close)

    def callculate_strategy_gain(self,date, signal, close):
        leverage = 1
        print_log = False
        result = pd.Series(np.nan, dtype='float64')
        start_balance = 1000
        balance = start_balance
        #print("balance:" + str(balance))
        last_sell_idx = -1
        last_buy_idx = -1
        for buy_idx in np.where(signal == 'buy')[0]:
            if last_buy_idx == -1 and buy_idx > last_sell_idx:
                last_buy_idx = buy_idx
                for sell_idx in np.where(signal == 'sell')[0]:
                    if last_buy_idx != -1 and sell_idx > buy_idx:
                        delta = (close[sell_idx] - close[buy_idx]) / close[buy_idx]
                        if (delta < -0.01):  # stop loss percentage
                            # stop loss
                            balance = balance + (balance * (delta * leverage))
                            balance = balance * 0.975
                            last_buy_idx = - 1
                            if print_log:
                                print(str(date[sell_idx]) + ' Stop Loss:' + str(balance) + ' -> delta:' + str(delta))

                        if sell_idx > last_buy_idx and last_buy_idx != - 1:
                            # profit sell
                            # print('Pair buy_idx :' + str(buy_idx) + ', sell_idx: ' + str(sell_idx))
                            # print('Pair buy :' + str(df['CLOSE'][buy_idx]) + ', sell: ' + str(df['CLOSE'][sell_idx]))
                            result[sell_idx] = close[sell_idx] - close[buy_idx]
                            balance = balance + (balance * (delta * leverage))
                            balance = balance * 0.975
                            last_sell_idx = sell_idx
                            last_buy_idx = -1
                            if print_log:
                                if delta > 0.025:
                                    print(str(date[sell_idx]) + ' Profit:' + str(balance) + ' -> delta:' + str(delta))
                                else:
                                    print(str(date[sell_idx]) + ' Loss:' + str(balance) + ' -> delta:' + str(delta))

        #print("final balance:" + str(balance))
        profit = (balance) / start_balance * 100
        if  profit < 100:
            profit = profit - 100
        return profit

    def plot_chart(self,chart_name, dataFrame, main_chart_indicators = []):
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
        
        fig.add_trace(plt.Candlestick(x     = dataFrame['Date'],
                                    open  = dataFrame['Open'], 
                                    high  = dataFrame['High'],
                                    low   = dataFrame['Low'], 
                                    close = dataFrame['Close'], name = chart_name), row=1, col=1)
        
        for indicator in main_chart_indicators:
            fig.add_trace(plt.Scatter(x=dataFrame['Date'], y=dataFrame[indicator], mode='lines', name= indicator), row=1, col=1)
        

        #fig.add_trace(plt.Scatter(x=dataFrame['Date'], y=dataFrame['MACD'], mode='lines', name='MACD'), row=3, col=1)
        #fig.add_trace(plt.Scatter(x=dataFrame['Date'], y=dataFrame['MACD_SIGNAL'], mode='lines', name='MACD_SIGNAL'), row=3, col=1)
        #fig.add_trace(plt.Bar(x = dataFrame['Date'], y= dataFrame['MACD_HIST'], name = 'MACD_HIST', marker= {'color': 'red'}), row=3, col=1)

        #fig.add_scatter(x=dataFrame['Date'], y=dataFrame['buy_price'], mode="markers", marker=dict(size=20), name="buy", row=1, col=1)
        #fig.add_scatter(x=dataFrame['Date'], y=dataFrame['sell_price'], mode="markers", marker=dict(size=20), name="sell", row=1, col=1)
        
        fig.update_layout(
            yaxis_title= chart_name
        )

        fig.show()

    def find_best_parameter(self,exchange, pair):
        for timeFrame in ['5m','15m','1h','1d']:
            for supertrend_period in range(1, 200, 1):
                for supertrend_factor in np.arange(1, 4, 1):
                    data = exchange.fetch_ohlcv(symbol = pair, timeframe = timeFrame, since=None, limit= 600)
                    header = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
                    df = pd.DataFrame(data, columns=header)#.set_index('Timestamp')
                    df['Timestamp'] /= 1000
                    df['Date'] = [datetime.fromtimestamp(x) for x in df['Timestamp']]
                    haDf = self.indicatorClass.HA(df)
                    self.indicatorClass.SuperTrend(df, supertrend_period, round(supertrend_factor, 1));
                    supertrend_signal = 'STX_' + str(supertrend_period) + '_' + str(round(supertrend_factor, 1))
                    profit = self.supertrend_strategy(df, supertrend_signal)
                    print(timeFrame + ';' + str(supertrend_period) + ';' + str(round(supertrend_factor, 1)) + ';' + str(profit))

    def limit_buy(self,exchange, api_key, api_secret, pair, amount,conf):

        self.buy(exchange, pair, amount,conf)

    def limit_sell(self,exchange, api_key, api_secret, pair, amount,conf):

        self.sell(exchange, pair, amount,conf)

    def main(self,exchange,conf,use_heikenashi = True):

        pair = conf['symbol']+"/"+conf['quote']

        #print(exchange.has)

        balance = exchange.fetchBalance()
        free_balance = balance['USDT']['free']
        print('free_balance: ' + str(free_balance) + ' USDT')
        #buy(exchange, pair, 20)
        #buy(exchange, pair, 100)
        #sell(exchange, pair, 90)

        #order = exchange.createLimitBuyOrder (pair, 0.5, 1.1)##15195897224
        #print(order)
        #print(order['id'])
        
        #order_book = exchange.fetchOrderBook(pair)
        #max_bid = order_book['bids'][0][0]
        #min_ask = order_book['asks'][0][0]
        #print(max_bid)
        #print(min_ask)
        #print(order_book)
        #exchange.cancelOrder(order['id'])
        #exchange.editOrder(id=15195897224, symbol=pair, type='limit', side='buy', amount=0.5, price=1.2)

        #print (exchange.fetchBalance())

        #if (exchange.has['fetchOrder']):
        #    print(exchange.fetchOrder (id=8539938604))
        #print(exchange.fetchOrders (pair))
        #print(exchange.fetchOpenOrders (pair))
        
        markets = exchange.loadMarkets()
        for market in markets:
            data = markets[market]
                
            #if data['symbol'].endswith(quote):
                #print( data['symbol'] + ' -> change1h  -> %' + str(float(data['info']['change1h'])*100))
                #print( data['symbol'] + ' -> change24h -> %' + str(float(data['info']['change24h'])*100))
                #print( data['symbol'] + ' -> changeBod -> %' + str(float(data['info']['changeBod'])*100))
        
        
        #self.find_best_parameter(exchange, pair)
        """
        """
        data = exchange.fetch_ohlcv(symbol = pair, timeframe = conf['timeFrame'], since=None, limit=conf['periot'])
        print("data price:", data[-1][4])
        if( conf['appendSubData']):
            subData = exchange.fetch_ohlcv(symbol = pair, timeframe = conf['subTimeFrame'], since=None, limit=60);
            subDataLastIndex= [x[0] for x in subData].index(data[-1][0])
            for x in subData[subDataLastIndex+1:]: 
                data.append(x)
        
        header = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = pd.DataFrame(data, columns=header)#.set_index('Timestamp')
        df['Timestamp'] /= 1000
        df['Date'] = [datetime.fromtimestamp(x) for x in df['Timestamp']]
        df['Symbol'] = pair
        df['Exchange'] = conf['exchangeId']
        filename = '{}.csv'.format(conf['timeFrame'])
        df = self.indicatorClass.HA(df, use_heikenashi)
        
        #self.indicatorClass.SuperTrend(df, conf['supertrendPeriod'], conf['supertrendFactor']);
        #df["T3"] = self.indicatorClass.generateTillsonT3(df, param=[conf['t3VolumeFactor'],conf['t3Period']])
        #SuperTrend(df, 11, 1);
      #  supertrend_signal = 'STX_{0}_{1}'.format(conf['supertrendPeriod'], conf['supertrendFactor'])
      #  supertrend_signal_price = 'ST_{0}_{1}'.format(conf['supertrendPeriod'], conf['supertrendFactor'])
      #  profit = self.supertrend_strategy(df, supertrend_signal)
      #  print('%' + str(profit))
      #  df = df[(df[supertrend_signal_price] > 0)]
      #  df = df.tail(150)
        #print(df.tail(50).to_string())
        #df = df.drop(df.index[[0,11]], inplace=True)
        #haDf = super_trend(haDf, 1, 10)
        #print(df.tail(25))
        #print(haDf.tail(25))
        if(conf['showGraph']):
            self.plot_chart(pair, df, [supertrend_signal_price, "T3","Signal","buy_price" ]);

        if(conf['indicator']=="t3"):
            tSignal = self.t3getSignal(df["T3"], conf,exchange)
        elif conf['indicator']=="supertred":
            tSignal = (df.tail(1))['Signal'].to_string()
        elif conf["indicator"]=="wave":
            tSignal = self.indicatorClass.wave(data)
        elif conf["indicator"]=="waveWT1":
            tSignal = self.indicatorClass.waveWT1(data)
        print(f"using {conf['indicator']}  signal")
        print("signal",tSignal)
        return tSignal
    def t3getSignal(self,df,conf,exchange):
        status = "None"
        pair = conf['symbol']+"/"+conf['quote']
        if(df[df.index[-1]]>df[df.index[-2]] and df[df.index[-2]]<df[df.index[-3]] and self.lastSignalT3!='buy'):
            status = "buy"
        if(df[df.index[-1]]<df[df.index[-2]] and df[df.index[-2]]>df[df.index[-3]] and self.lastSignalT3!='sell'):
            status = "sell"
        if(status!="None" ):
            now = datetime.now()
            date = now.strftime("%m/%d/%Y, %H:%M:%S")
            with open('output/'+conf['exchangeId']+'_'+conf['symbol']+'_T3_'+str(conf['timeFrame'])+'.csv', 'a+', newline='') as file:
                fieldnames = ['coinName','type', 'price','date']
                writer = DictWriter(file, fieldnames=fieldnames)
                writer.writerow({'coinName':conf['symbol'],'type': status, 'price': exchange.fetchOrderBook(pair)['bids'][0][0],'date':date})
        return status
        
    def start(self,conf,confFile):
        exchange = getattr(ccxt, conf['exchangeId'])({
            'enableRateLimit': True, 
            'apiKey': conf['apiKey'],
            'secret': conf['apiSecret']
        })
        if(conf['subaccount']):
            exchange.headers = {
                'FTX-SUBACCOUNT': 'test',
            }
        pair = conf['symbol']+"/"+conf['quote']
        while True:
            try:
                signal = self.main(exchange,conf)
                self.exchange(signal,exchange,conf,confFile,pair);
                
                if( conf['showGraph']):
                    break
                else:
                    sleep(conf['repeatSecond'])
            except:
                print("sel buy problem")    


    def exchange(self,signal,exchange,conf,confFile,pair):
        try:
            if(signal.find("buy")>-1 and self.lastSignal!='buy'):
                self.lastSignal = "buy"
                if(conf['test']==False):
                    self.limit_buy(exchange, conf['apiKey'], conf['apiSecret'], pair, exchange.fetch_balance()[conf['quote']]['free'],conf)
                self.writeLog(signal, exchange,conf,confFile,pair)
            if(signal.find("sell")>-1 and self.lastSignal!='sell'):
                self.lastSignal = "sell"
                if(conf['test']==False):
                    self.limit_sell(exchange, conf['apiKey'], conf['apiSecret'], pair, exchange.fetch_balance()[conf['symbol']]['free'],conf)
                self.writeLog(signal, exchange,conf,confFile,pair)
        except:
            print("sel buy problem")
        
    def writeLog(self,signal,exchange,conf,confFile,pair):
        now = datetime.now()
        date = now.strftime("%m/%d/%Y, %H:%M:%S")
        with open('output/'+conf['exchangeId']+'_'+confFile+'_'+conf['symbol']+'_sellbuycalculation_'+conf['indicator']+"_"+str(conf['timeFrame'])+'.csv', 'a+', newline='') as file:
                fieldnames = ['coinName','type', 'price','date']
                writer = DictWriter(file, fieldnames=fieldnames)
                writer.writerow({'coinName':conf['symbol'],'type': signal[len(signal)-4:len(signal)], 'price': exchange.fetchOrderBook(pair)['bids'][0][0],'date':date})
    def writeLogSelBuy(self,signal,pair,price,conf,exchange,free):
        now = datetime.now()
        date = now.strftime("%m/%d/%Y, %H:%M:%S")
        with open('output/market_SellBuy'+conf['exchangeId']+'_'+conf['indicator']+"_"+str(conf['timeFrame'])+'.csv', 'a+', newline='') as file:
                fieldnames = ['coinName','type', 'price','date',"free"]
                writer = DictWriter(file, fieldnames=fieldnames)
                writer.writerow({'coinName':pair,'type': signal, 'price': str(price),'date':date,'free':free})

