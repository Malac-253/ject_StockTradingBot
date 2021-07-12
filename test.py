# db_manager.py

#By Malachi I Beerram
#GitHub @malac-253
#Started:       06/23/21
#Last Updated:  07/01/21

## Arguments Check
import sys
print(f"Arguments count: {len(sys.argv)}")
for i, arg in enumerate(sys.argv):
    print(f"Argument {i:>6}: {arg}")

##Imports
import pandas_datareader as web
from datetime import datetime
import pandas as pd
import csv
import psycopg2
import os
import matplotlib
from sqlalchemy import create_engine
from tqdm import tqdm
from datetime import date
import random
import traceback
import numpy as np
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
from pandas_datareader._utils import RemoteDataError
import investpy
import requests
import trendet
import time
print("Imports loaded",'\n')

#BASE_DIR
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
## Create a database connection
db_password = "CSC280" # database password
db_name = "stockdata" # Name of database
fileName = BASE_DIR +"/src/stock.txt" # Gets stocks from file

##START
print("Creating the standard table-base")

## Creating database engine for working with
engine = create_engine('postgresql://postgres:{}@localhost/{}'.format(db_password,db_name))

## setting dates for date (YYYY,MM,DD)
start = int(time.mktime(datetime(2011,1,1,23,59).timetuple()))
end = int(time.mktime(datetime(2021,1,1,23,59).timetuple()))
interval = '3d'

## Gets stocks from file
stocks = ['GOOG']
print("Stocks to get :",stocks,end="\n")

## Removing old table if EXISTS
all_resampled = ['daily','weekly','monthly','quarterly']

## Getting data for each stock, based in symbol
for symbol in tqdm(stocks, desc='Creating Stock tables',position=1):
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1={start}&period2={end}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query_string)


    print(query_string)
    print(df)

    ## Will manage backtesting 
def backtesting():
    all_resampled = ['daily','weekly','monthly']
    start = '2018-12-31 00:00:00'
    end = '2020-01-01 00:00:00'

    query = """
            SELECT * FROM _"""+all_resampled[0]+"""_prices 
                    WHERE symbol = 'NSS' 
                    AND date > '""" +start+"""' 
                    AND date < '""" +end+"""' 
                ORDER BY symbol ASC, date ASC
                """
    data = engine.execute(query)
    dataframe = pd.DataFrame(data.fetchall())
    dataframe.columns = data.keys()
    print(dataframe.head())

    # list of parameters which are configurable for the strategy
    class SmaCross(bt.Strategy):
        params = dict(
            pfast=10,  # period for the fast moving average
            pslow=30   # period for the slow moving average
        )

        def __init__(self):
            sma1 = bt.ind.SMA(period=self.p.pfast)  # fast moving average
            sma2 = bt.ind.SMA(period=self.p.pslow)  # slow moving average
            self.crossover = bt.ind.CrossOver(sma1, sma2)  # crossover signal

        def next(self):
            if not self.position:  # not in the market
                if self.crossover > 0:  # if fast crosses slow to the upside
                    self.buy()  # enter long

            elif self.crossover < 0:  # in the market & cross to the downside
                self.close()  # close long position

    cerebro = bt.Cerebro()  # create a "Cerebro" engine instance

    # Pass it to the backtrader datafeed and add it to the cerebro
    data = bt.feeds.PandasData(dataname=dataframe)

    cerebro.broker.set_cash(100)
    cerebro.broker.setcommission(commission=0.0001)

    cerebro.adddata(data)  # Add the data feed

    cerebro.addstrategy(SmaCross)  # Add the trading strategy

    print('Starting Portfolio Value : %0.2f' % cerebro.broker.getvalue())
    cerebro.run() # run it all
    # cerebro.plot() # and plot it with a single command

    print('Final Portfolio Value : %0.2f' % cerebro.broker.getvalue())