# db_manager.py

#By Malachi I Beerram
#GitHub @malac-253
#Started:		06/23/21
#Last Updated:	07/19/21

## Arguments Check
import sys
print(">db_manager.py")
print(f"Arguments count: {len(sys.argv)}")
for i, arg in enumerate(sys.argv):
    print(f"Argument {i:>6}: {arg}")

print("Imports loading ...")

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
import backtrader as bt
from plotly.subplots import make_subplots
import json
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
from itertools import permutations

print("Imports loaded  :)")

#BASE_DIR
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
## Create a database connection
db_password = "CSC280" # database password
db_name = "stockdata" # Name of database
fileName = BASE_DIR +"/src/stock.txt" # Gets stocks from file
engine = create_engine('postgresql://postgres:{}@localhost/{}'.format(db_password,db_name))## Creating database engine for working with
all_resampled = [['daily','weekly','monthly'],['1d', '1wk', '1mo']] ## periods to wotk with

## Will create the initial database for the initial list of stocks.
def managetables():
	##START
	
	## Note: turn in to function
	## Gets stocks from file
	fileObj = open(fileName, "r") #opens the file in read mode
	stocks = fileObj.read().splitlines() #puts the file into an array 
	fileObj.close() #closes the file
	print("Stocks to get :",stocks,end="\n")## stocks = ['GOOG',"AAPL","MMM"]

	start = 0 # assigned in managetables if statment
	end = 1 # assigned in managetables if statment

	## managetables if statment 
	if sys.argv[1]=='createtables':
		print("Creating the standard table-base")
		## setting dates for date (YYYY,MM,DD)
		start = int(time.mktime(datetime(2011,1,1,23,59).timetuple()))
		end = int(time.mktime(datetime(2021,1,1,23,59).timetuple()))

		## Removing old table if EXISTS
		for samp in tqdm(all_resampled[0], desc='Cleaning database',position=0):
			query = 'DROP TABLE IF EXISTS _'+samp+'_prices;'
			engine.execute(query)
	elif sys.argv[1]=='updatetables':
		print("Updating the standard table-base")

		start = datetime(2020,1,1,23,59)
		## Sets today as end time, or sets arg as end time
		end = date.today()

		## Adding in commandline Argument for Start and end
		if len(sys.argv) == 4 and not sys.argv[2]=='end' and not sys.argv[2]=='start':
			start = sys.argv[2].split('-')
			print( start)
			if int(sys.argv[2].split('-')[0]) < 2004 :
				print("\n","Can't go futher back than 2004, sorry")
				print("Will start  with year 2004")
				start = datetime(int(start[0]),int(start[1]),int(start[2]))
			start = datetime(2004,int(start[1]),int(start[2]))

			end = sys.argv[3].split('-')
			end = datetime(int(end[0]),int(end[1]),int(end[2]))

		if len(sys.argv) == 4 and sys.argv[2]=='end':
			end = sys.argv[3].split('-')
			if int(sys.argv[3].split('-')[0]) < 2004 :
				print("\n","Can't go futher back than 2004, sorry")
				print("Will start  with year 2004")
				end = datetime(int(start[0]),int(start[1]),int(start[2]))
			else:
				end = datetime(2004,int(start[1]),int(start[2]))
		elif len(sys.argv) == 4 and sys.argv[2]=='start':
			start = sys.argv[3].split('-')
			if int(sys.argv[3].split('-')[0]) < 2004 :
				print("\n","Can't go futher back than 2004, sorry")
				print("Will start  with year 2004")
				start = datetime(int(start[0]),int(start[1]),int(start[2]))
			else:
				start = datetime(2004,int(start[1]),int(start[2]))

		## Sets as start day to look for data
		start = int(time.mktime(start.timetuple()))
		## Sets today as end time, or sets arg as end time
		end = int(time.mktime(end.timetuple()))
	else:print("Something might be Wrong")


	## Getting data for each stock, based in symbol
	for symbol in tqdm(stocks, desc='Managing Stock data >> ' + sys.argv[1],position=1):
		all_df = []

		## Trying to get the data from finance.yahoo
		try:
			for interval in all_resampled[1]:
				query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1={start}&period2={end}&interval={interval}&events=history&includeAdjustedClose=true'
				df = pd.read_csv(query_string)
				all_df.append(df)
		except KeyError:
			print("\n-KeyError-")
			traceback.print_exc()
			print("\n Most likely "+symbol+" has to little data for tables-base \n\t--No data saved")
		except Exception:
			traceback.print_exc()
			print("\nSomething else unexpected went wrong - No data saved")
			print("Make sure you actually maded the database, you want to add data to")
			print("Make sure the password is correct")
		except RemoteDataError as exp :
			print("RemoteDataError - > All attempts to get data have failed")
			traceback.print_exc()
	
		## manage SQL input if statment 
		if sys.argv[1]=='createtables':
			## Creating tables with data
			create_prices_table(symbol,all_df)
		elif sys.argv[1]=='updatetables':
			## Updateing tables with data
			update_prices_table(symbol,all_df)


	## Create a primary key on the table
	if sys.argv[1]=='createtables':
		for samp in tqdm(all_resampled[0], desc='Introducing integrity to database',position=0):
			query = 'ALTER TABLE _'+samp+'_prices ADD PRIMARY KEY (symbol, date);'
			engine.execute(query)

	##END
	return 'Prices for all table created'

## Create a SQL table directly from a dataframe
def create_prices_table(symbol,all_df):
	##START
	counter = 0
	for df in tqdm(all_df, desc='Processing the resampled data for '+symbol,position=0):
		## Processing the data
		df = processing_data(df,symbol)
		##Convert to SQL DataBase
		df.to_sql('_'+all_resampled[0][counter]+"_prices", engine, if_exists='append', index=False)
		counter = counter+1
	#END
	return 'Prices for '+symbol.lower()+' table created'

## update the SQL table directly from a dataframe of new data	
def update_prices_table(symbol,all_df):
	##START

	## end of the large query
	## Handle duplicate values - Avoiding errors if you've already got some data in your table
	insert_end = """ ON CONFLICT (symbol, date) DO UPDATE 
			SET
            volume = EXCLUDED.volume,
            open = EXCLUDED.open,
            close = EXCLUDED.close,
            high = EXCLUDED.high,
            low = EXCLUDED.low;
            """

	counter = 0
	for df in tqdm(all_df, desc='Processing the new resampled data for ' + symbol,position=0):
		## Processing the data
		df = processing_data(df,symbol)

		## First part of the insert statement
		insert_init = """ INSERT INTO _"""+all_resampled[0][counter]+"""_prices  
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

		## Put together the query string
		query = insert_init + vals + insert_end
    
    	## Fire insert statement
		engine.execute(query)
		counter = counter+1

	return 'Prices for '+symbol+' updated'
	##END

## Processing the raw dataframe date
def processing_data(df,symbol):
	##START

	## Processing the data
	df = df.fillna(0) ## Fill in to Make sure all data is full
	df = df.rename(str.lower, axis='columns') ## editing col names to be lowercase
	df.drop(columns=(list(df.columns)[5]), axis=1, inplace=True) ##Removing unwanteddata
	for row in df['date']: datetime.strptime(row, '%Y-%m-%d') ## Converting ti time date Object
	df = df.set_index(pd.DatetimeIndex(df['date']))## Setting as index
	df.drop(columns=(list(df.columns)[0]), axis=1, inplace=True) ##Removing unwanteddata
	df['updated'] = pd.to_datetime('now')## Add Updata timestamp
	df.insert(0, "date",df.index[...], True) ## Adding in data of date
	df.insert(0, "symbol",symbol, True) ## Adding in data of sysmbol id
	df['volume'] = df['volume'].apply(np.int64) ## insureing that 'volume' is an int

	return df
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

		# ## set dates if an end and a start are picked 
		# if len(sys.argv) > 5 and sys.argv[4] != "all":
		# 	query_start_date = " AND date > '" + (sys.argv[4]) + " 00:00:00'"
		# 	file_name = file_name + "D" + sys.argv[4].replace('-', 's')+ "_"

		# if len(sys.argv) > 5 and sys.argv[5] != "all":
		# 	query_end_date = " AND date < '" + (sys.argv[5]) + " 00:00:00'"
		# 	file_name = file_name + "_" + sys.argv[5].replace('-', 's')+ "_"

		# ## set dates if an end or a start are picked
		# if len(sys.argv) > 5 and sys.argv[4] == "start":
		# 	query_start_date = " AND date < '" + (sys.argv[5]) + " 00:00:00'"
		# 	query_end_date 	= ""
		# 	file_name = symbol + "_" + "D" + sys.argv[5].replace('-', 's') + "_End" + "_"
		
		# if len(sys.argv) > 5 and sys.argv[4] == "end":
		# 	query_end_date = " AND date > '" + (sys.argv[5]) + " 00:00:00'"
		# 	query_start_date 	= ""
		# 	file_name = symbol + "_" + "D" + "Start_" + sys.argv[5].replace('-', 's') + "_"

		query = query_int + query_start_date + query_end_date + query_fin

		## Fetches the daily prices of the stocks and pust them into a dataframe
		data = engine.execute(query)
		df = pd.DataFrame(data.fetchall())
		df.columns = data.keys()

		# ## reduces cols based on time type
		df = df
		file_name = file_name + "daily"

		# if sys.argv[3] == "day":
		# 	df = df
		# 	file_name = file_name + "daily"
		# elif sys.argv[3] == "week":print("[Abandoned] (Will default to day) See Help file 'db_manager.py help' ")
		# 	# dataTemp = []
		# 	# for row in df.itertuples():
		# 	# 	if row[2].weekday() == 0:
		# 	# 		print(row[2])
		# 	# 		if row[0]+4 < len(df.index):
		# 	# 			print("-",df.iat[(row[0]+4),1],"\n")
		# elif sys.argv[3] == "biweek":print("[Abandoned] (Will default to day) See Help file 'db_manager.py help' ")
		# elif sys.argv[3] == "month":print("[Abandoned] (Will default to day) See Help file 'db_manager.py help' ")
		# elif sys.argv[3] == "3month":print("[Abandoned] (Will default to day) See Help file 'db_manager.py help' ")
		# elif sys.argv[3] == "year":print("[Abandoned] (Will default to day) See Help file 'db_manager.py help' ")
		# elif sys.argv[3] == "all":print("[Abandoned] (Will default to day) See Help file 'db_manager.py help' ")


		## Creating the figure
		fig = go.Figure(data=[go.Candlestick(x=df['date'],
	                open=df['open'],
	                high=df['high'],
	                low=df['low'],
	                close=df['close'])])

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
		if (sys.argv[6] == "save" and len(sys.argv) > 6) or  (sys.argv[6] == "both" and len(sys.argv) > 6):
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
				fig.write_image(BASE_DIR +"/src/charts/raw/png/"+file_name+".png")

		if (sys.argv[6] == "show" and len(sys.argv)) or  (sys.argv[6] == "both" and len(sys.argv)) > 6:
			print ("Attempting to show ...")
			fig.show()

















# ## start working with other custom Indicator analysis
# 	## Full Wick average lengths for data
# 		wick_avg_full_len 	= []; wick_avg_full_len_helper 	= 0 
# 		wick_avg_high_len 	= []; wick_avg_high_len_helper 	= 0
# 		wick_avg_low_len	= []; wick_avg_low_len_helper 	= 0
# 	## Trending markets
# 		lowest_low 			= []; lowest_low_helper			= 100000 	;
# 		lowest_high			= []; lowest_high_helper		= 100000 	;
# 		highest_high 		= []; highest_high_helper 		= 0 		;
# 		highest_low 		= []; highest_low_helper 		= 0 		;
# 		trend_find = False #False to find low,True to find high

# 		for index, row in tqdm(dfmain.iterrows(), desc='Working on '+samp+' for '+symbol, position=0):
# 		## Checking amount of space between the dates and adding answer to list that many time
# 		##	To fill in list and dates acturaly 
			
# 			if (index + 1) < len(dfmain['date']):
# 				times = dates.index(dfmain.iloc[index+1]['date'])-dates.index(dfmain.iloc[index]['date'])
	
# 	## start working with data
# 		## Full Wick average lengths for data
# 			# wick_avg_day_full_len
# 			wick_len_temp = (row[2] - row[3]) - abs((row[4] - row[5])) 
# 			wick_avg_full_len_helper = wick_avg_full_len_helper + wick_len_temp
# 			for x in range(times):wick_avg_full_len.append(wick_avg_full_len_helper/(index+1))
# 			# wick_avg_day_high_len
# 			wick_len_temp = (row[2] - max(row[4],row[5]))
# 			wick_avg_high_len_helper = wick_avg_high_len_helper + wick_len_temp
# 			for x in range(times):wick_avg_high_len.append(wick_avg_high_len_helper/(index+1))
# 			# wick_avg_day_low_len
# 			wick_len_temp = (min(row[4],row[5]) - row[3])
# 			wick_avg_low_len_helper = wick_avg_low_len_helper + wick_len_temp
# 			for x in range(times):wick_avg_low_len.append(wick_avg_low_len_helper/(index+1))
# 		## Trending markets
# 			#if row[3] > lowest_low_helper:

# 			if trend_find: # looking for highest_high
# 				highest_high_helper = max(highest_high_helper,row[2])
# 				for x in range(times):highest_high.append(highest_high_helper)
# 				trend_find = False
# 			else: # looking for lowest_low
# 				lowest_low_helper = min(lowest_low_helper,row[3])
# 				for x in range(times):lowest_low.append(lowest_low_helper)
# 				trend_find = True

# 	## Full Wick average lengths for data
# 		all_data_ans.append(wick_avg_full_len)
# 		colum_text.append('fullwick_averagelen_'+samp)
# 		all_data_ans.append(wick_avg_high_len)
# 		colum_text.append('highwick_averagelen_'+samp)
# 		all_data_ans.append(wick_avg_low_len)
# 		colum_text.append('lowwick_averagelen_'+samp)
# 	## Trending markets
# 		#lowest_low
# 		all_data_ans.append(lowest_low)
# 		colum_text.append('lowest_low_'+samp)
# 		#highest_high
# 		all_data_ans.append(highest_high)
# 		colum_text.append('highest_high_'+samp)


# 	# awful code but it's late and this works

# 	#date_string='24/10/2011 12:43'
# 	time_add= []
	
# 	list1 = [1, 2]
# 	list2 = [1, 2, 3, 4, 5, 6, 7, 8]
# 	for x in range(4):
# 		time_ducktapefix = '0'+str(random.choice(list2))+'/'+str(random.choice(list1))+str(random.choice(list2))+'/2000 01:01'
# 		time_add.append(datetime.strptime(time_ducktapefix,'%m/%d/%Y %H:%M'))
# 	# Create DataFrame
# 	df = pd.DataFrame(all_data_ans)
# 	df = df.transpose()
# 	df.columns = colum_text
# 	df.insert(0, "symbol",symbol, True) ## Adding in data of sysmbol id 
# 	df.insert(1, "date",dates + time_add, True)	
# 	df = df.fillna(0) ## Fill in to Make sure all data is full
# 	#print(samp,'\n',df.head(),'\n',df.tail(),'\n')
# 	df.to_sql("_test_analysis_table_prices", engine, if_exists='append', index=False)


# query = 'ALTER TABLE _test_analysis_table_prices ADD PRIMARY KEY (symbol, date);'
# engine.execute(query)
	













		# initialize list of lists

		
		


		

		

		# # Create the pandas DataFrame
		# #df = pd.DataFrame(data, columns = ['Name', 'Age'])
		# df = pd.DataFrame(data, columns = columns)
		# print(df)

## Will manage backtesting 
def backtesting():
	all_resampled = ['daily','weekly','monthly']
	start = '2004-12-31 00:00:00'
	end = '2021-07-08 00:00:00'
	stock = 'MBI'

	query = """
			SELECT * FROM _test_analysis_table_prices 
					WHERE symbol = '"""+stock+"""' 
					AND date > '""" +start+"""' 
					AND date < '""" +end+"""' 
				ORDER BY symbol ASC, date ASC
				"""
	data = engine.execute(query)
	dataframe = pd.DataFrame(data.fetchall())
	dataframe.columns = data.keys()

	class DummyInd(bt.Indicator):
		lines = ('dummyline',)
		params = (('value', 10),)

		def next(self):
			self.lines.dummyline[0] = max(0.0, self.params.value)

		def once(self, start, end):
			self.lines.dummyline.array = dataframe['simplemovingaverage_5'].tolist()
	
	class sma5(bt.Indicator):
		lines = ('simplemovingaverage_5',)

		params = (('num',0),)#(('value', 5),)

		#print(bt.Indicator)
		def __init__(self):
			self.simplemovingaverage_5 = dataframe['simplemovingaverage_5']
			#self.lines.simplemovingaverage_5 = 
			print(self.simplemovingaverage_5)


	dataframe.set_index('date', inplace=True)

	# list of parameters which are configurable for the strategy
	class SmaCross(bt.Strategy):
		params = dict(
			pfast=10,  # period for the fast moving average
			pslow=30   # period for the slow moving average
		)

		def __init__(self):

			#self.smatest = DummyInd(value=1)
			
			self.sma1 = bt.ind.MovingAverageBase(period=20)
			#bt.ind.Ichimoku(tenkan=9,kijun=26,senkou=52,senkou_lead=26,chikou=26)

			self.sma1.plotinfo.subplot=False
			self.sma2 = bt.ind.SMA(period=20)
			self.sma2.plotinfo.subplot=False
			#print(self.sma1)



			# print(sma1)
			#self.sma2 = bt.ind.fractal(period=5,bardist=0.015,shift_to_potential_fractal=2)  # slow moving average
			#self.sma2 = bt.studies.Fractal()

			#print(self.data.date)
			#self.counter = 0
			# print(self.sma1)
			# print(self.sma2)
			# accde1 = bt.ind.MomentumOsc(period= 12,band= 100)  # slow moving average
			# accde2 = bt.ind.MomentumOsc(period= 100,band= 100)  # slow moving averag
			#self.crossover1 = bt.ind.CrossOver(self.sma1, self.sma2)  # crossover signal
			self.crossover2 = bt.ind.CrossOver(self.sma1, self.sma2)  # crossover signal
			# print(self.crossover1)

			# self.crossover2 = bt.ind.CrossOver(accde1, accde2)  # crossover signal

		def next(self):
			if not self.position:  # not in the market
				if self.crossover2 > 0:  # if fast crosses slow to the upside
					self.buy()  # enter long

			elif self.crossover2 < 0:  # in the market & cross to the downside
				self.close()  # close long position
			# if self.counter == 30:
			# 	if not self.position:  # not in the market
			# 		self.buy()  # enter long
			# 		self.counter = 0

			# 	else:  # in the market & cross to the downside
			# 		self.close()  # close long position
			# 		self.counter = 0

			

		# def stop(self):
		# 	myvalues = self.mysma.sma.get(size=len(self.mysma))
		# 	for val in myvalues:
		# 		print(val)

	cerebro = bt.Cerebro()  # create a "Cerebro" engine instance
	# cerebro = bt.Cerebro(stdstats=False)

	# Pass it to the backtrader datafeed and add it to the cerebro
	data = bt.feeds.PandasData(dataname=dataframe)

	cerebro.broker.set_cash(1000)
	cerebro.broker.setcommission(commission=0.01)
	cerebro.adddata(data)  # Add the data feed
	cerebro.addstrategy(SmaCross)  # Add the trading strategy



	cerebro.addsizer(bt.sizers.PercentSizer, percents=20)  # default sizer for strategies



	print('Starting Portfolio Value : %0.2f' % cerebro.broker.getvalue())
	cerebro.run() # run it all
	cerebro.plot() # and plot it with a single command

	print('Final Portfolio Value : %0.2f' % cerebro.broker.getvalue())

	cerebro.addwriter(bt.WriterFile, csv=True, out='test2.csv')

	##df = pd.read_csv(cerebro.addwriter(bt.WriterFile, csv=True, out='test.csv'))

	##print(df.head())



def geneticalgorithm():
	print("Starting Main Genetic Algorithm Running ... this could take a while")

	## Getting all information needed
	start = '2004-12-31 00:00:00'
	end = '2021-07-08 00:00:00'
	stock = 'MBI'

	query = """
			SELECT * FROM _test_analysis_table_prices 
					WHERE symbol = '"""+stock+"""' 
					AND date > '""" +start+"""' 
					AND date < '""" +end+"""' 
				ORDER BY symbol ASC, date ASC
			"""

	data = engine.execute(query)
	dataframe = pd.DataFrame(data.fetchall())
	dataframe.columns = data.keys()
	dataframe.set_index('date', inplace=True)

	##START


	float_safety_prim = 10000
	plot_spacing = 1#5

	def Price_based_Indicators_lines_helper_coverter(PD_indicator_object,
		Parameters_dic,
		Temp_lines_list):
		## Calculating procedurally-generated name
		Indicator_name = PD_indicator_object['Indicator_str'].replace('ind','').lower()+'_'+str("_".join(map(str, Parameters_dic.values()))).replace('.','p')
		## Setting data name as a line name for lines in backtrader
		Temp_lines_list.append(Indicator_name)
	def Price_based_Indicators_lines_json_helper_coverter(PD_indicator_object,
		Link_Indicator_Parameters_dic,
		Curr_Indicator_Parameters_dic,
		Indicator_Parameters_keys,
		Temp_lines_list):

		## If value is a float then will move to reasonable integer but multiplying by float_safety_prim (some number with alot of zeros)
		float_safety = 1
		if isinstance(Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][0], float):
			float_safety = float_safety_prim
		if isinstance(Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][1], float):
			float_safety = float_safety_prim
		if isinstance(Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][2], float):
			float_safety = float_safety_prim

		## Will set the min(start), max(end), and amount(freqs) for intra-indicator indicator parameters formations
		start = Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][0]*float_safety
		end = Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][1]*float_safety
		freqs = Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][2]*float_safety

		## Creating or adding to curr set of indicator parameters, to make indicator parameters formation
		for x in range(int(start),int(end),int(freqs)):
			if (float_safety == 1) and not (x == 0):
				Curr_Indicator_Parameters_dic[Indicator_Parameters_keys[0]] = int(x)
			elif x == 0:
				Curr_Indicator_Parameters_dic[Indicator_Parameters_keys[0]] = int(x)
			else:
				Curr_Indicator_Parameters_dic[Indicator_Parameters_keys[0]] = x/float_safety

			## if more parameters founded, recursively adding another layer to the indicator parameters formation
			if len(Indicator_Parameters_keys) > 1:
				Price_based_Indicators_lines_json_helper_coverter(PD_indicator_object,
					Link_Indicator_Parameters_dic,
					Curr_Indicator_Parameters_dic,
					Indicator_Parameters_keys[1:],
					Temp_lines_list)
			else:
				Price_based_Indicators_lines_helper_coverter(PD_indicator_object,Curr_Indicator_Parameters_dic,Temp_lines_list)
	def Price_based_Indicators_lines_coverter(Price_based_Indicators):
		temp_lines = []
		for PD_indicator_object in Price_based_Indicators:
			if len(list(PD_indicator_object['Parameters_dic'].keys())) > 0 :
				Price_based_Indicators_lines_json_helper_coverter(PD_indicator_object, 	#PD_indicator_object
					PD_indicator_object['Parameters_dic'], 								#Link_Indicator_Parameters_dic
					{}, 																#Curr_Indicator_Parameters_dic
					list(PD_indicator_object['Parameters_dic'].keys()),					#Indicator_Parameters_keys
					temp_lines)															#Temp_lines_list 	
			else:
				Price_based_Indicators_lines_helper_coverter(PD_indicator_object['Indicator_str'],{},temp_lines)
		return temp_lines

	## The Indicator analysis array
	Price_based_Indicators = json.load(open('Price_based_Indicators.json'))# Opening JSON file
	Price_based_Indicators_lines = Price_based_Indicators_lines_coverter(Price_based_Indicators)
	#print(Price_based_Indicators_lines)
	## takes data from database and makes it in to backtrader usable object of Indicator type
	## Helper fuctions
	def Price_based_Indicators_helper_coverter(Indicator_object,
		PD_indicator_object,
		Parameters_dic):
		## Calculating procedurally-generated name
		Indicator_name = PD_indicator_object['Indicator_str'].replace('ind','').lower()+'_'+str("_".join(map(str, Parameters_dic.values()))).replace('.','p')
		## Setting data to it's line in backtrader
		setattr(getattr(getattr(Indicator_object,"lines"),Indicator_name),"array",dataframe[Indicator_name].tolist())
	def Price_based_Indicators_json_helper_coverter(Indicator_object,
		PD_indicator_object,
		Link_Indicator_Parameters_dic,
		Curr_Indicator_Parameters_dic,
		Indicator_Parameters_keys):

		## If value is a float then will move to reasonable integer but multiplying by float_safety_prim (some number with alot of zeros)
		float_safety = 1
		if isinstance(Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][0], float):
			float_safety = float_safety_prim
		if isinstance(Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][1], float):
			float_safety = float_safety_prim
		if isinstance(Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][2], float):
			float_safety = float_safety_prim

		## Will set the min(start), max(end), and amount(freqs) for intra-indicator indicator parameters formations
		start = Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][0]*float_safety
		end = Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][1]*float_safety
		freqs = Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][2]*float_safety

		## Creating or adding to curr set of indicator parameters, to make indicator parameters formation
		for x in range(int(start),int(end),int(freqs)):
			if (float_safety == 1) and not (x == 0):
				Curr_Indicator_Parameters_dic[Indicator_Parameters_keys[0]] = int(x)
			elif x == 0:
				Curr_Indicator_Parameters_dic[Indicator_Parameters_keys[0]] = int(x)
			else:
				Curr_Indicator_Parameters_dic[Indicator_Parameters_keys[0]] = x/float_safety

			## if more parameters founded, recursively adding another layer to the indicator parameters formation
			if len(Indicator_Parameters_keys) > 1:
				Price_based_Indicators_json_helper_coverter(Indicator_object,
					PD_indicator_object,
					Link_Indicator_Parameters_dic,
					Curr_Indicator_Parameters_dic,
					Indicator_Parameters_keys[1:])
			else:
				Price_based_Indicators_helper_coverter(Indicator_object,PD_indicator_object,Curr_Indicator_Parameters_dic)
	## Main Class
	class Price_based_database_converter(bt.Indicator):
		lines = tuple(Price_based_Indicators_lines)
		def once(self, start, end):
			for PD_indicator_object in tqdm(Price_based_Indicators, desc='Database Data converter ', position=0):
				if len(list(PD_indicator_object['Parameters_dic'].keys())) > 0 :
					Price_based_Indicators_json_helper_coverter(self,			#Indicator_object (database_converter Indicator_object)				
						PD_indicator_object, 									#PD_indicator_object
						PD_indicator_object['Parameters_dic'], 					#Link_Indicator_Parameters_dic
						{}, 													#Curr_Indicator_Parameters_dic
						list(PD_indicator_object['Parameters_dic'].keys()))		#Indicator_Parameters_keys)												 	
				else:
					Price_based_Indicators_helper_coverter(self,PD_indicator_object['Indicator_str'],{})

	## Hold the Strategy applied to the test based on the chromosme
	class chromosome_StrategyContainer(bt.Strategy):
		def __init__(self):
			self.price_based_data = Price_based_database_converter()
			self.price_based_data.plotinfo.subplot=False
			self.price_based_data.plotinfo.plotlinevalues=True

			# print("heer")
			
			# print(self)

			## Method 1
			""" 
			Basic crossover of all the price similar types of Indicators, for examples all the types of avagers
			"""
			basic_price_gauge_crossover(self, status ='__init__')

			#self.smatest = DummyInd(value=1)

		# def next(self):
		# 	if not self.position:  # not in the market
		# 		self.buy()  # enter long

		# 	else:
		# 		self.close()  
	
	def basic_price_gauge_crossover(self,status='__init__'):
		gene_list = []
		def basic_price_gauge_crossover_init(self):
			crossover_groups =list(permutations(Price_based_Indicators_lines, 2))
			print(len(crossover_groups))
			for crossover_group in tqdm(crossover_groups, desc='basic_price_gauge_crossover_init', position=0):
				gene_name = ("crossover"+("_".join(crossover_group).replace("_","")))
				gene_list.append(gene_name)

				## Creating crossover genes
				setattr	(self, # Object to set attr to
						gene_name, # attr name
						bt.ind.CrossOver( # Set values for attr
							getattr(getattr(self.price_based_data,"lines"),crossover_group[0]),
							getattr(getattr(self.price_based_data,"lines"),crossover_group[1])
							) 
						)
			##calls bt.'*bt_src'.'*Indicator_str'(*Parameters_dic)
			##ex: calls bt.ind.SimpleMovingAverage(**Parameters_dic)
			#)

				#self.

		def basic_price_gauge_crossover_next(self):
			print("ba")

		if status == '__init__': basic_price_gauge_crossover_init(self)
		elif status == 'next': basic_price_gauge_crossover_next(self)



	cerebro = bt.Cerebro()  # create a "Cerebro" engine instance
	# cerebro = bt.Cerebro(stdstats=False)

	# Pass it to the backtrader datafeed and add it to the cerebro
	data = bt.feeds.PandasData(dataname=dataframe)

	cerebro.broker.set_cash(1000)
	cerebro.broker.setcommission(commission=0.01)
	cerebro.adddata(data)  # Add the data feed
	cerebro.addstrategy(chromosome_StrategyContainer)  # Add the trading strategy



	cerebro.addsizer(bt.sizers.PercentSizer, percents=20)  # default sizer for strategies



	print('Starting Portfolio Value : %0.2f' % cerebro.broker.getvalue())
	cerebro.run() # run it all
	cerebro.plot() # and plot it with a single command
	print('Final Portfolio Value : %0.2f' % cerebro.broker.getvalue())

def custom_analysis(plot_figure_obj,main_dataframe):
    def support_resistance_detection(plot_figure_obj,df):
        ## Credit: https://towardsdatascience.com/detection-of-price-support-and-resistance-levels-in-python-baedc44c34c9
        def isSupport(df,i):
          support = df['low'][i] < df['low'][i-1]  and df['low'][i] < df['low'][i+1] and df['low'][i+1] < df['low'][i+2] and df['low'][i-1] < df['low'][i-2]
          return support
        def isResistance(df,i):
          resistance = df['high'][i] > df['high'][i-1]  and df['high'][i] > df['high'][i+1] and df['high'][i+1] > df['high'][i+2] and df['high'][i-1] > df['high'][i-2]
          return resistance
        def isFarFromLevel(l,levels,s):
           return np.sum([abs(l-x) < s  for x in levels]) == 0

        def collapse_levels(df,levels,s):
            new_levels =  [[levels[0][0],levels[0][1],levels[0][0],levels[0][1],levels[0][0],levels[0][1]]]

            for level in levels:
                mark = False

                for idx, le in enumerate(new_levels):
                    if abs(le[1]-level[1]) < s:
                        # print('work',level[1],le[1])
                        tec = [level[0],level[1],le[0],min(level[1],le[1])*0.95,level[0],max(level[1],le[1])*1.05]  #mx,my,x1,y1,x2,y2
                        new_levels.pop(idx)
                        new_levels.insert((idx-1),tec)
                    else:
                        mark = True
                if mark:
                    new_levels.append([level[0],level[1],level[0],level[1]*0.95,level[0],level[1]*1.05])#mx,my,x1,y1,x2,y2


            res = []
            for i in new_levels:
                if i not in res:
                    res.append(i)

            return res


        def plot_all(fig,df,levels):
            # print(df.head())
          # fig, ax = plt.subplots()

          # candlestick_ohlc(ax,df.values,width=0.6, \
          #                  colorup='green', colordown='red', alpha=0.8)
          # date_format = mpl_dates.DateFormatter('%d %b %Y')
          # ax.xaxis.set_major_formatter(date_format)
          # fig.autofmt_xdate()
          # fig.tight_layout()
            y_sp = 30
            for level in levels:
                #print(df['date'][level[0]])
                #print(level[1])
                while level[4]+1 > len(df['date']):
                    level[4] = level[4]-1

                while level[4]+y_sp+1 > len(df['date']):
                    y_sp = y_sp-1
                
                fig.add_trace(
                    go.Scatter( x=[(df['date'][level[2]]),  (df['date'][level[2]]),     (df['date'][level[4]+y_sp]),    (df['date'][level[4]+y_sp])], 
                                y=[(level[3]),              (level[5]),         (level[5]),         (level[3])],
                                fill="toself"
                        )
                    ,row=1, col=1) 

        ## START

        s =  np.mean(df['high'] - df['low'])*0.5

        levels = []
        for i in range(2,df.shape[0]-2):
          if isSupport(df,i):
            levels.append((i,df['low'][i]))
          elif isResistance(df,i):
            levels.append((i,df['high'][i]))

        # levels = []
        # for i in range(2,df.shape[0]-2):
        #   if isSupport(df,i):
        #     l = df['low'][i]
        #     if isFarFromLevel(l,levels,s):
        #       levels.append((i,l))
        #   elif isResistance(df,i):
        #     l = df['high'][i]
        #     if isFarFromLevel(l,levels,s):
        #       levels.append((i,l))

        # print(levels)
        new_levels = collapse_levels(main_dataframe,levels,s)
        #print(new_levels)
        plot_all(plot_figure_obj,main_dataframe,new_levels)

        ## END









    support_resistance_detection(plot_figure_obj,main_dataframe)

## Handles Arg and useage	
def main_manager():
	if(sys.argv[1]=='help' and len(sys.argv) > 1):print(open("db_manager_help.txt", "r").read())
	elif(sys.argv[1]=='createtables'and len(sys.argv) > 1):managetables()
	elif(sys.argv[1]=='updatetables' and len(sys.argv) > 1):managetables()
	elif(sys.argv[1]=='chart' and len(sys.argv) < 9):chart() 
	elif(sys.argv[1]=='analysis' and len(sys.argv) <= 10):analysis() 
	elif(sys.argv[1]=='backtesting' and len(sys.argv) <= 10):backtesting()
	elif(sys.argv[1]=='geneticalgorithm' and len(sys.argv) <= 10):geneticalgorithm()
	else:print("Incorrect/Unrecognized Arguments, Type 'db_manager.py help' for help")

if __name__ == "__main__":
	print("BASE_DIR = ",BASE_DIR)
	print("Today_Da = ",pd.to_datetime('now'),'\n')
	main_manager()

