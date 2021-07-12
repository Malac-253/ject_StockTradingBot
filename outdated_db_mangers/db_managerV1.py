# main.py
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

## Will create the initial database for the initial list of stocks.
def createdatabase(arg):

	## Create a database connection
	db_password = "CSC280" # Set to your own password
	if len(arg.argv) > 3: 
		print("Creating a specific database")
		print(f"Arguments count: {len(sys.argv)}")
		for i, arg in enumerate(sys.argv):
			print(f"Argument {i:>6}: {arg}")
		db_password = arg.argv[3]

	db_name = "stockdata" # put in your own name
	if len(arg.argv) > 4: db_db_name = arg.argv[4]
	engine = create_engine('postgresql://postgres:{}@localhost/{}'.format(db_password,db_name))

	##path to data (to place data)
	data_path = 'historical_data'

 	## (YYYY,MM,DD)
	start = datetime(2011,1,1)
	end = datetime(2021,1,1)

	## Gets stocks from file
	#stocks = ['GOOG',"AAPL","MMM"]
	fileName = 'stock.txt'
	if len(arg.argv) > 2: fileName = arg.argv[2]

	fileObj = open(fileName, "r") #opens the file in read mode
	stocks = fileObj.read().splitlines() #puts the file into an array
	fileObj.close()

	print("(DB_Builder) Stocks to get :",end="\n")
	print(stocks,end="\n")

	for stock in tqdm(stocks, desc='Creating stock database ...'):
		try:
			df = web.DataReader(stock,'yahoo',start,end)
			df.to_csv(f'historical_data/stockdata_{stock}.csv')
			create_prices_table(stock,df,engine)
		except KeyError:
			print("\n"+stock+" has to little data for database \n\t-- missing between 2011,1,1 - 2021,1,1 \n\t--No data saved")
		except:
			print("Something else unexpected went wrong - No data saved")
		

	return '(DB_Builder)-- >> Daily prices for all table created'

# Create a SQL table directly from a dataframe
def create_prices_table(symbol,df,engine):
	##START

	## Fill in to Make sure all data is full
	df = df.fillna(0) 

	##editing col names
	df = df.rename(str.lower, axis='columns')

	##Removing unwanteddata
	df.drop(columns=(list(df.columns)[5]), axis=1, inplace=True)

	## Add Updata timestamp
	df['updated'] = pd.to_datetime('now')
	
	##Adding in Data of date
	df.insert(0, "date",df.index[...], True)
	df.insert(0, "symbol",symbol, True)
	
	
	##Convert to SQL DataBase
	df.to_sql((symbol.lower()+'_daily_prices'), engine, if_exists='replace', index=False)

	## Create a primary key on the table
	query = 'ALTER TABLE '+symbol.lower()+'_daily_prices ADD PRIMARY KEY (symbol, date);'
	engine.execute(query)

	##END
	return '(DB_Builder)-- >> Daily prices for '+symbol.lower()+' table created'

## Will create the initial database for the initial list of stocks.
def updatedatabase(arg):

	## Create a database connection
	db_password = "CSC280" # Set to your own password
	if len(arg.argv) > 5: 
		# print("Create a specific database")
		# print(f"Arguments count: {len(sys.argv)}")
		# for i, arg in enumerate(sys.argv):
		# 	print(f"Argument {i:>6}: {arg}")
		db_password = arg.argv[5]

	db_name = "stockdata" # put in your own name
	if len(arg.argv) > 6: db_db_name = arg.argv[6]
	engine = create_engine('postgresql://postgres:{}@localhost/{}'.format(db_password,db_name))

	##path to data (to place data)
	data_path = 'historical_data'

	## Gets stocks from file
	#stocks = ['GOOG',"AAPL","MMM"]
	fileName = 'stock.txt'
	if len(arg.argv) > 4: fileName = arg.argv[4]

	fileObj = open(fileName, "r") #opens the file in read mode
	stocks = fileObj.read().splitlines() #puts the file into an array
	fileObj.close()

	print("Stocks to get :",end="\n")
	print(stocks,end="\n")
	if len(arg.argv) == 4:
		if int(arg.argv[2].split('-')[0]) < 2005 :
			print("\n","Can't go futher back than 2005, sorry")
			print("Will start  with year 2005")

	for stock in tqdm(stocks, desc='Updating stock database ...'):
		## Query database for the daily_prices
		query = 'SELECT * FROM '+stock.lower()+'_daily_prices ;'
		intel = engine.execute(query)
		
		## Fetchs the last daily_prices (the top row for database)
		row = intel.fetchone()

		## Sets as start day to look for data
		start = datetime(row[1].year,row[1].month,row[1].day)
		
		## Sets today as end time, or sets arg as end time
		end = date.today()

		##Adding in commandline Argument for Start and end
		if len(arg.argv) == 2:
			##Do nonting
			end = date.today()
		elif len(arg.argv) == 4:
			##Getting Start day
			start = arg.argv[2].split('-')
			start = datetime(int(start[0]),int(start[1]),int(start[2]))
			if int(arg.argv[2].split('-')[0]) < 2005 :
				start = arg.argv[2].split('-')
				start = datetime((2005),int(start[1]),int(start[2]))

			end = arg.argv[3].split('-')
			end = datetime(int(end[0]),int(end[1]),int(end[2]))
		else :
			raise UpdateDataDaseArgumentError

		df = web.DataReader(stock,'yahoo',start,end)
		df = df.fillna(0) 
		##Adding in Data of date
		df.insert(0, "date",df.index[...], True)
		df.insert(0, "symbol",stock, True)

 		## First part of the insert statement
		insert_init = 'INSERT INTO '+stock.lower()+"""_daily_prices  
				(symbol, date, high, low, open, close, volume, updated) VALUES """

		## Add values for all days to the insert statement
		vals = ",".join(["""('{}', '{}', '{}', '{}', '{}', '{}', '{}','{}')""".format(
		                     row['symbol'],
		                     row['date'],
		                     row['High'],
		                     row['Low'],
		                     row['Open'],
		                     row['Close'],
		                     row['Volume'],
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

	return '(DB_Builder)-- >> Daily prices for all table created'
	



## Handles Arg and useage	
def main_manager(arg):
	if(arg.argv[1]=='help'):print(open("db_manager_help.txt", "r").read())
	elif(arg.argv[1]=='createdatabase'):createdatabase(arg)
	elif(arg.argv[1]=='createspecificdatabase'):createdatabase(arg)
	elif(arg.argv[1]=='updatedatabase'):updatedatabase(arg)
	else:print("Incorrect/Unrecognized Arguments >>> Type 'db_manager.py help' for help")


##Must have 2 or 0 input not one
class UpdateDataDaseArgumentError(Exception):
	def __init__(self, message="updatedatabase must have 2 Arguments or 0"):
		self.message = message
		super().__init__(self.message)
#pass

if __name__ == "__main__":
    # print(f"Arguments count: {len(sys.argv)}")
    # for i, arg in enumerate(sys.argv):
    #     print(f"Argument {i:>6}: {arg}")
    main_manager(sys)

