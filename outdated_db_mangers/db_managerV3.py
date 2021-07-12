# db_manager.py

#By Malachi I Beerram
#GitHub @malac-253
#Started:		06/23/21
#Last Updated:	07/01/21

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
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta

#BASE_DIR
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
## Create a database connection
db_password = "CSC280" # database password
db_name = "stockdata" # Name of database
fileName = BASE_DIR +"/src/stock.txt" # Gets stocks from file

## Will create the initial database for the initial list of stocks.
def createtables():
	##START
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

	## Removing old table if EXISTS
	query = 'DROP TABLE IF EXISTS _daily_prices;'
	engine.execute(query)

	## Getting data for each stock, based in symbol
	for symbol in tqdm(stocks, desc='Creating Stock tables'):
		try:
			df = web.DataReader(symbol,'yahoo',start,end)
			create_prices_table(symbol,df,engine)
		except KeyError:
			print("-KeyError-")
			print("\n Most likely "+symbol+" has to little data for tables-base \n\t-- missing data between 2011,1,1 - 2021,1,1 \n\t--No data saved")
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
	print("Updating a specific table-base")

	## Creating database engine for working with
	engine = create_engine('postgresql://postgres:{}@localhost/{}'.format(db_password,db_name))

	## Gets stocks from file
	## stocks = ['GOOG',"AAPL","MMM"]
	fileObj = open(fileName, "r") #opens the file in read mode
	stocks = fileObj.read().splitlines() #puts the file into an array 
	fileObj.close() #closes the file
	print("Stocks to get :",stocks,end="\n")

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

## Will create a basic candle stick chart of data using plotly
def chart():
	## python db_manager.py chart JYNT,GOOG day 2021-01-01 2021-06-01 save html
	## Creating database engine for working with
	engine = create_engine('postgresql://postgres:{}@localhost/{}'.format(db_password,db_name))

	## Getting symbolsto chart
	symbols = sys.argv[2].split(",")

	if symbols[0] == "all":
		data = engine.execute("SELECT symbol FROM _daily_prices GROUP BY symbol")
		symbols = [item['symbol'] for item in data.fetchall()]
		
	print("Stocks to chart",symbols)

	for symbol in tqdm(symbols, desc='Creating Chart Stock'):
		file_name = symbol + "_"

		## Query database for the daily_prices
		query_int = "SELECT * FROM _daily_prices WHERE symbol = '" +symbol+"'"
		query_start_date = ""
		query_end_date = ""
		query_fin = " ORDER BY symbol ASC, date ASC"

		## set dates if an end and a start are picked 
		if len(sys.argv) > 5 and sys.argv[4] != "all":
			query_start_date = " AND date > '" + (sys.argv[4]) + " 00:00:00'"
			file_name = file_name + "D" + sys.argv[4].replace('-', 's')+ "_"

		if len(sys.argv) > 5 and sys.argv[5] != "all":
			query_end_date = " AND date < '" + (sys.argv[5]) + " 00:00:00'"
			file_name = file_name + "_" + sys.argv[5].replace('-', 's')+ "_"

		## set dates if an end or a start are picked
		if len(sys.argv) > 5 and sys.argv[4] == "start":
			query_start_date = " AND date < '" + (sys.argv[5]) + " 00:00:00'"
			query_end_date 	= ""
			file_name = symbol + "_" + "D" + sys.argv[5].replace('-', 's') + "_End" + "_"
		
		if len(sys.argv) > 5 and sys.argv[4] == "end":
			query_end_date = " AND date > '" + (sys.argv[5]) + " 00:00:00'"
			query_start_date 	= ""
			file_name = symbol + "_" + "D" + "Start_" + sys.argv[5].replace('-', 's') + "_"

		query = query_int + query_start_date + query_end_date + query_fin

		## Fetches the daily prices of the stocks and pust them into a dataframe
		data = engine.execute(query)
		df = pd.DataFrame(data.fetchall())
		df.columns = data.keys()

		# ## reduces cols based on time type
		if sys.argv[3] == "day":
			df = df
			file_name = file_name + "daily"
		elif sys.argv[3] == "week":print("[Abandoned] (Will default to day) See Help file 'db_manager.py help' ")
			# dataTemp = []
			# for row in df.itertuples():
			# 	if row[2].weekday() == 0:
			# 		print(row[2])
			# 		if row[0]+4 < len(df.index):
			# 			print("-",df.iat[(row[0]+4),1],"\n")
		elif sys.argv[3] == "biweek":print("[Abandoned] (Will default to day) See Help file 'db_manager.py help' ")
		elif sys.argv[3] == "month":print("[Abandoned] (Will default to day) See Help file 'db_manager.py help' ")
		elif sys.argv[3] == "3month":print("[Abandoned] (Will default to day) See Help file 'db_manager.py help' ")
		elif sys.argv[3] == "year":print("[Abandoned] (Will default to day) See Help file 'db_manager.py help' ")
		elif sys.argv[3] == "all":print("[Abandoned] (Will default to day) See Help file 'db_manager.py help' ")


		## Creating the figure
		fig = go.Figure(data=[go.Candlestick(x=df['date'],
	                open=df['open'],
	                high=df['high'],
	                low=df['low'],
	                close=df['close'])])
		fig.update_layout()

		fig.update_layout(
			xaxis_rangeslider_visible=False,
		    title= 'Daily Candlestick Chart of '+ symbol,
		    yaxis_title='Stock Price',
		    xaxis_title='Date'
		)
		#     shapes = [dict(
		#         x0='2016-12-09', x1='2016-12-09', y0=0, y1=1, xref='x', yref='paper',
		#         line_width=2)],
		#     annotations=[dict(
		#         x='2016-12-09', y=0.05, xref='x', yref='paper',
		#         showarrow=False, xanchor='left', text='Increase Period Begins')]
		# )

		## Manageing save and view options
		if (sys.argv[6] == "save" and len(sys.argv)) or  (sys.argv[6] == "both" and len(sys.argv)) > 6:
			if not os.path.exists(BASE_DIR +"/src/charts"):os.mkdir(BASE_DIR +"/src/charts")
			if not os.path.exists(BASE_DIR +"/src/charts/raw"):os.mkdir(BASE_DIR +"/src/charts/raw")
			
			do = "all"
			if len(sys.argv) == 8:
				do = sys.argv[7]

			if do == "html" or do == "all":
				if not os.path.exists(BASE_DIR +"/src/charts/raw/html"):os.mkdir(BASE_DIR +"/src/charts/raw/html")
				fig.write_html(BASE_DIR +"/src/charts/raw/html/"+file_name+".html")

			if do == "svg" or do == "all":
				if not os.path.exists(BASE_DIR +"/src/charts/raw/svg"):os.mkdir(BASE_DIR +"/src/charts/raw/svg")
				fig.write_image(BASE_DIR +"/src/charts/raw/svg/"+file_name+".svg")

			if do == "png" or do == "all":
				if not os.path.exists(BASE_DIR +"/src/charts/raw/png"):os.mkdir(BASE_DIR +"/src/charts/raw/png")
				fig.write_image(BASE_DIR +"/src/charts/raw/svg/"+file_name+".png")

		if (sys.argv[6] == "show" and len(sys.argv)) or  (sys.argv[6] == "both" and len(sys.argv)) > 6:
			print ("Attempting to show ...")
			fig.show()

## Will create the tables of daily price calculated data featuring all the variables and identifiers needed for analysis.  
def analysis():
	engine = create_engine('postgresql://postgres:{}@localhost/{}'.format(db_password,db_name))

	## Getting symbolsto chart
	symbols = sys.argv[2].split(",")

	if symbols[0] == "all":
		data = engine.execute("SELECT symbol FROM _daily_prices GROUP BY symbol")
		symbols = [item['symbol'] for item in data.fetchall()]
		
	print("Stocks to chart",symbols)








## Handles Arg and useage	
def main_manager():
	if(sys.argv[1]=='help' and len(sys.argv) > 1):print(open("db_manager_help.txt", "r").read())
	elif(sys.argv[1]=='createtables'and len(sys.argv) > 1):createtables()
	elif(sys.argv[1]=='createtablesspecific' and len(sys.argv) == 4):print("createtablesspecific is [deprecated]" )
	elif(sys.argv[1]=='updatetables' and len(sys.argv) > 1):updatetables()
	elif(sys.argv[1]=='updatetablesspecific'and len(sys.argv) == 7):print("updatetablesspecific is [deprecated]" )
	elif(sys.argv[1]=='chart' and len(sys.argv) < 9):chart() 
	else:print("Incorrect/Unrecognized Arguments, Type 'db_manager.py help' for help")

if __name__ == "__main__":
    print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        print(f"Argument {i:>6}: {arg}")
    print("BASE_DIR = ",BASE_DIR)
    print("Today_Da = ",pd.to_datetime('now'),'\n')
    main_manager()

