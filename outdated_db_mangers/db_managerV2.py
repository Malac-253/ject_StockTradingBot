# db_manager.py

#By Malachi I Beerram
#Started : 6/23/21
#Last Updated : 6/26/21

import pandas_datareader as web
from datetime import datetime
import pandas as pd
import csv
import psycopg2
import os
import matplotlib
from sqlalchemy import create_engine
from tqdm import tqdm
import sys
from datetime import date
import random
import traceback
import numpy as np


## Will create the initial database for the initial list of stocks.
def createtables():
	##START

	## Create a database connection
	db_password = "CSC280" # database password
	db_name = "stockdata" # Name of database
	fileName = 'stock.txt' # Gets stocks from file

	## Getting password and name if specific database
	if (sys.argv[1] == 'createtablesspecific'): 
		print("Creating a specific table-base")
		db_password = sys.argv[3]
		db_name = sys.argv[4]
		fileName = sys.argv[2]
	else:
		print("Creating the standard table-base")

	## Creating database engine for working with
	engine = create_engine('postgresql://postgres:{}@localhost/{}'.format(db_password,db_name))

 	## setting dates for date (YYYY,MM,DD)
	start = datetime(2011,1,1)
	end = datetime(2021,1,1)

	## Gets stocks from file
	## stocks = ['GOOG',"AAPL","MMM"]
	fileObj = open(fileName, "r") #opens the file in read mode
	stocks = fileObj.read().splitlines() #puts the file into an array 
	fileObj.close() #closes the file
	print("Stocks to get :",stocks,end="\n")

	## Create master list of all stocks
	dfmaster = pd.DataFrame(stocks, columns = ['symbol'])
	dfmaster.insert(1, "updated",pd.to_datetime('now'), True)
	dfmaster.to_sql(('_stocks_symbol_master_list'), engine, if_exists='replace', index=False)
	## Create a primary key on the table
	query = 'ALTER TABLE _stocks_symbol_master_list ADD PRIMARY KEY (symbol);'
	engine.execute(query)

	## Removing old table if EXISTS
	query = 'DROP TABLE IF EXISTS _daily_prices;'
	engine.execute(query)

	## Getting data for each stock, based in symbol
	for symbol in tqdm(stocks, desc='Creating Stock tables'):
		try:
			df = web.DataReader(symbol,'yahoo',start,end)
			df.to_csv(f'historical_data/stockdata_{symbol}.csv')
			create_prices_table(symbol,df,engine)
		except KeyError:
			print("\n"+symbol+" has to little data for tables-base \n\t-- missing between 2011,1,1 - 2021,1,1 \n\t--No data saved")
		except Exception:
			traceback.print_exc()
			print("Something else unexpected went wrong - No data saved")
			print("Make sure you actually maded the database, you want to add data to")
			print("Make sure the password is correct")
	
	## Create a primary key on the table
	query = 'ALTER TABLE _daily_prices ADD PRIMARY KEY (symbol, date);'
	engine.execute(query)

	##END
	return 'Daily prices for all table created'

## Create a SQL table directly from a dataframe
def create_prices_table(symbol,df,engine):
	##START

	## Processing the data
	df = df.fillna(0) ## Fill in to Make sure all data is full
	df = df.rename(str.lower, axis='columns') ## editing col names to be lowercase
	df.drop(columns=(list(df.columns)[5]), axis=1, inplace=True) ##Removing unwanteddata
	df['updated'] = pd.to_datetime('now')## Add Updata timestamp
	df.insert(0, "date",df.index[...], True) ## Adding in data of date
	df.insert(0, "symbol",symbol, True) ## Adding in data of sysmbol id
	df['volume'] = df['volume'].apply(np.int64)
	##Convert to SQL DataBase
	df.to_sql('_daily_prices', engine, if_exists='append', index=False)

	##END
	return '(DB_Builder)-- >> Daily prices for '+symbol.lower()+' table created'

## Will create the initial database for the initial list of stocks.
def updatetables():
	##START

	## Create a database connection
	db_password = "CSC280" # database password
	db_name = "stockdata" # Name of database
	fileName = 'stock.txt' ## Gets stocks from file

	## Getting password and name if specific database
	if (sys.argv[1] == 'updatetablesspecific'): 
		print("Updating a specific table-base")
		db_password = sys.argv[3]
		db_name = sys.argv[4]
		fileName = sys.argv[2]# Getting fileName if specific file
	else:
		print("Updating the standard table-base")

	## Creating database engine for working with
	engine = create_engine('postgresql://postgres:{}@localhost/{}'.format(db_password,db_name))

	## Gets stocks from file
	## stocks = ['GOOG',"AAPL","MMM"]
	fileObj = open(fileName, "r") #opens the file in read mode
	stocks = fileObj.read().splitlines() #puts the file into an array 
	fileObj.close() #closes the file
	print("Stocks to get :",stocks,end="\n")

	## Updating master list of all stocks
	dfmaster = pd.DataFrame(stocks, columns = ['symbol'])
	insert_init_master = """INSERT INTO _stocks_symbol_master_list (symbol, updated) VALUES """
	## Add values for the new stocks to the insert statement
	vals_master = ",".join(["""('{}', '{}')""".format(
	                     row['symbol'],
						 pd.to_datetime('now')
	                     ) for date, row in dfmaster.iterrows()])
	## Handle duplicate values
	insert_end_master = """ ON CONFLICT (symbol) DO UPDATE SET
            		updated = EXCLUDED.updated;"""
	query = insert_init_master + vals_master + insert_end_master ## Put together the query string
	engine.execute(query) ## Fire insert statement


	## UPDATEING the datatable
	if len(sys.argv) == 4:
		if int(sys.argv[2].split('-')[0]) < 2005 :
			print("\n","Can't go futher back than 2005, sorry")
			print("Will start  with year 2005")

	if (sys.argv[1] == 'updatetablesspecific'): 
		if int(sys.argv[6].split('-')[0]) < 2005 :
			print("\n","Can't go futher back than 2005, sorry")
			print("Will start  with year 2005")

	for symbol in tqdm(stocks, desc='Updating stock database ...'):
		update_prices_table(symbol,engine)
	
	##END
	return 'Daily prices for all table updated'

	## Create a SQL table directly from a dataframe

## update the SQL table directly from a dataframe of new data	
def update_prices_table(symbol,engine):
	##START

	## Query database for the daily_prices
	query = "SELECT * FROM _daily_prices WHERE symbol = '"+symbol+"' ORDER BY date DESC"
	## Fetchs the last daily_prices (the top row for database)
	row = engine.execute(query).fetchone()
	
	##Setting time period to start from.
	start = datetime(row[1].year,row[1].month,row[1].day)## Sets as start day to look for data
	end = date.today()## Sets today as end time, or sets arg as end time

	##Adding in commandline Argument for Start and end
	if len(sys.argv) == 4 and not sys.argv[1] == 'updatetablesspecific':
		##Getting Start day
		start = sys.argv[2].split('-')
		start = datetime(int(start[0]),int(start[1]),int(start[2]))

		if int(sys.argv[2].split('-')[0]) < 2005 :
			start = sys.argv[2].split('-')
			start = datetime((2005),int(start[1]),int(start[2]))

		end = sys.argv[3].split('-')
		end = datetime(int(end[0]),int(end[1]),int(end[2]))
	elif sys.argv[1] == 'updatetablesspecific':
		##Getting Start day
		start = sys.argv[6].split('-')
		start = datetime(int(start[0]),int(start[1]),int(start[2]))

		if int(sys.argv[6].split('-')[0]) < 2005 :
			start = sys.argv[6].split('-')
			start = datetime((2005),int(start[1]),int(start[2]))

		end = sys.argv[7].split('-')
		end = datetime(int(end[0]),int(end[1]),int(end[2]))
	

	df = web.DataReader(symbol,'yahoo',start,end)
	## Processing the data
	df = df.fillna(0) ## Fill in to Make sure all data is full
	df = df.rename(str.lower, axis='columns') ## editing col names to be lowercase
	##Adding in Data of date
	df.insert(0, "date",df.index[...], True)
	df.insert(0, "symbol",symbol, True)

		## First part of the insert statement
	insert_init = """ INSERT INTO _daily_prices  
			(symbol, date, high, low, open, close, volume, updated) VALUES """

	## Add values for all days to the insert statement
	vals = ",".join(["""('{}', '{}', '{}', '{}', '{}', '{}', '{}','{}')""".format(
	                     row['symbol'],
	                     row['date'],
	                     row['high'],
	                     row['low'],
	                     row['open'],
	                     row['close'],
	                     int(row['volume']),
						 pd.to_datetime('now')
	                     ) for date, row in df.iterrows()])

	## Handle duplicate values - Avoiding errors if you've already got some data in your table
	insert_end = """ ON CONFLICT (symbol, date) DO UPDATE 
			SET
            volume = EXCLUDED.volume,
            open = EXCLUDED.open,
            close = EXCLUDED.close,
            high = EXCLUDED.high,
            low = EXCLUDED.low;
            """

    ## Put together the query string
	query = insert_init + vals + insert_end
    
    ## Fire insert statement
	engine.execute(query)
	
	return 'Daily prices for '+symbol+' updated'
	##END

## Will create a basic candle stick chart of data using matplotlib
def chart():

	## Create a database connection
	db_password = "CSC280" # database password
	db_name = "stockdata" # Name of database
	fileName = 'stock.txt' ## Gets stocks from file

	## Creating database engine for working with
	engine = create_engine('postgresql://postgres:{}@localhost/{}'.format(db_password,db_name))

## Will create the tables of daily price calculated data featuring all the variables and identifiers needed for analysis.  
def createinteldatabase():
	## getting the data
	if len(sys.argv) > 3 :
		if(sys.argv[2]=='help'):print(open("db_createinteldatabase_help.txt", "r").read())
		elif(sys.argv[3]=='all'):print("all variables and identifiers for :" + sys.argv[2])
		elif(sys.argv[3]=='trends'):intel_trends(defaults = 1, symbol = sys.argv[2])
		else:print("Unrecognized 2nd Argument, Type 'db_manager.py createinteldatabase help' for help")
	elif len(sys.argv) == 3:
		print("all variables and identifiers for :" + sys.argv[2])
	else:
		print("all stocks and all variables and identifiers")

def intel_trends(defaults = 0, symbol = "all"):#all market trends
	print("Up and Down trends for :" + symbol)
	print(defaults)





## Handles Arg and useage	
def main_manager():
	if(sys.argv[1]=='help' and len(sys.argv) > 1):print(open("db_manager_help.txt", "r").read())
	elif(sys.argv[1]=='createtables'and len(sys.argv) > 1):createtables()
	elif(sys.argv[1]=='createtablesspecific' and len(sys.argv) == 4):createtables()
	elif(sys.argv[1]=='updatetables' and len(sys.argv) > 1):updatetables()
	elif(sys.argv[1]=='updatetablesspecific'and len(sys.argv) == 7):updatetables()
	elif(sys.argv[1]=='chart' and len(sys.argv) <5):chart() #stock,days
	else:print("Incorrect/Unrecognized Arguments, Type 'db_manager.py help' for help")

if __name__ == "__main__":
    print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        print(f"Argument {i:>6}: {arg}")
    main_manager()

