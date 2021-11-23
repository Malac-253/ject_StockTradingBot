# average_based_indicators_analysis.py

#By Malachi I Beerram
#GitHub @malac-253
#Started:		07/19/21
#Last Updated:	07/19/21

## Arguments Check
import sys
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
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
from itertools import permutations
from itertools import combinations_with_replacement
from itertools import product
from itertools import chain
import gc
import time
import pprint
from functools import partial
from multiprocessing import Pool

#BASE_DIR
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
## Create a database connection
db_password = "CSC280" # database password
db_name = "stockdata" # Name of database
fileName = BASE_DIR +"/src/stock.txt" # Gets stocks from file
engine = create_engine('postgresql://postgres:{}@localhost/{}'.format(db_password,db_name))## Creating database engine for working with

## Getting symbolsto chart
symbols = sys.argv[2].split(",")

#if all getting all from database
if symbols[0] == "all":
    data = engine.execute("SELECT symbol FROM _daily_prices GROUP BY symbol")
    symbols = [item['symbol'] for item in data.fetchall()]

start = '2004-12-31 00:00:00'
end = '2021-07-08 00:00:00'


gene_spacer_main_MT = 2


## Will create the tables of daily price calculated data featuring all the variables and identifiers needed for analysis.  
"""
The current system is extremely inefficient however it's inefficiency is purposeful to allow me to more 
easily circumventing the back trader indicator line strategy pattern, to allow more ease-of-use while 
implementing the genetic algorithm with less computional usage
"""
def average_based_indicators_analysis():
    ##START
    print("Creating the analysis table-base ... this could take a while")

    ## Getting symbolsto chart
    symbols = sys.argv[2].split(",")

    #if all getting all from database
    if symbols[0] == "all":
        data = engine.execute("SELECT symbol FROM _daily_prices GROUP BY symbol")
        symbols = [item['symbol'] for item in data.fetchall()]
    print("Stocks for analysis :",symbols)

    float_safety_prim = 10000
    plot_spacing = 1#5
    ## Indicator analysis helper, handle Indicator stock analysis object
    def Indicator_analysis_helper_init(Indicator_analysis_obj,Indicator_object,Parameters_dic):
        ## Calculating procedurally-generated name
        Indicator_name = Indicator_object['Indicator_str'].replace('ind','').lower()+'_'+str("_".join(map(str, Parameters_dic.values()))).replace('.','p')
        ## Creating base Indicator attribute
        setattr(Indicator_analysis_obj, Indicator_name, 
            getattr(getattr(bt,Indicator_object['bt_src']),
            Indicator_object['Indicator_str'])(**Parameters_dic) 
            ##calls bt.'*bt_src'.'*Indicator_str'(*Parameters_dic)
            ##ex: calls bt.ind.SimpleMovingAverage(**Parameters_dic)
        )
    def Indicator_analysis_helper_stop(Indicator_analysis_obj,Indicator_object,Parameters_dic):
        ## Calculating procedurally-generated name
        Indicator_name = Indicator_object['Indicator_str'].replace('ind','').lower()+'_'+str("_".join(map(str, Parameters_dic.values()))).replace('.','p')

        ##Fetching base Indicator attribute and it's data and adding to dataframe
        Ind_tmp = getattr(Indicator_analysis_obj, Indicator_name)
        dfmain[Indicator_name] = Ind_tmp.get(size=len(Ind_tmp))
    def Indicator_analysis_helper_plot(Plot_figure_obj,Indicator_object,Parameters_dic,
        Weighted_squeeze_flo = 1,
        Plot_row_int = 1 ,
        Plot_col_int = 1, 
        Mean_factor_bol = False):
        ## Calculating procedurally-generated name
        Indicator_name = Indicator_object['Indicator_str'].replace('ind','').lower()+'_'+str("_".join(map(str, Parameters_dic.values()))).replace('.','p')

        ## adding view scaling if needed
        mean = 1
        if Mean_factor_bol : mean = dfmain['close'].mean()

        ## adding a trace line to the to the Plot_figure_obj 
        Plot_figure_obj.add_trace(
                go.Scatter(
                    x=dfmain.index[...],
                    y=((dfmain[Indicator_name]*Weighted_squeeze_flo)*(mean)),
                    name=Indicator_name),
                row=Plot_row_int, col=Plot_col_int
            ) 
    ## Using a json to support easy Indicator adding - these recurvis based on Indicator parameters 
    def Indicator_analysis_helper_json_init(Indicator_analysis_obj,
        Indicator_object,
        Link_Indicator_Parameters_dic,
        Curr_Indicator_Parameters_dic,
        Indicator_Parameters_keys):

        float_safety = 1
        if isinstance(Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][0], float):
            float_safety = float_safety_prim
        if isinstance(Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][1], float):
            float_safety = float_safety_prim
        if isinstance(Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][2], float):
            float_safety = float_safety_prim

        start = Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][0]*float_safety
        end = Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][1]*float_safety
        freqs = Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][2]*float_safety

        for x in range(int(start),int(end),int(freqs)):
            if (float_safety == 1) and not (x == 0):
                Curr_Indicator_Parameters_dic[Indicator_Parameters_keys[0]] = int(x)
            elif x == 0:
                Curr_Indicator_Parameters_dic[Indicator_Parameters_keys[0]] = int(x)
            else:
                Curr_Indicator_Parameters_dic[Indicator_Parameters_keys[0]] = x/float_safety

            if len(Indicator_Parameters_keys) > 1:
                Indicator_analysis_helper_json_init(Indicator_analysis_obj,Indicator_object,
                    Link_Indicator_Parameters_dic,
                    Curr_Indicator_Parameters_dic,
                    Indicator_Parameters_keys[1:])
            else:
                Indicator_analysis_helper_init(Indicator_analysis_obj,Indicator_object,Curr_Indicator_Parameters_dic)
    def Indicator_analysis_helper_json_stop(Indicator_analysis_obj,
        Indicator_object,
        Link_Indicator_Parameters_dic,
        Curr_Indicator_Parameters_dic,
        Indicator_Parameters_keys):

        float_safety = 1
        if isinstance(Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][0], float):
            float_safety = float_safety_prim
        if isinstance(Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][1], float):
            float_safety = float_safety_prim
        if isinstance(Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][2], float):
            float_safety = float_safety_prim

        start = Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][0]*float_safety
        end = Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][1]*float_safety
        freqs = Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][2]*float_safety

        for x in range(int(start),int(end),int(freqs)):
            if (float_safety == 1) and not (x == 0):
                Curr_Indicator_Parameters_dic[Indicator_Parameters_keys[0]] = int(x)
            elif x == 0:
                Curr_Indicator_Parameters_dic[Indicator_Parameters_keys[0]] = int(x)
            else:
                Curr_Indicator_Parameters_dic[Indicator_Parameters_keys[0]] = x/float_safety

            if len(Indicator_Parameters_keys) > 1:
                Indicator_analysis_helper_json_stop(Indicator_analysis_obj,Indicator_object,
                    Link_Indicator_Parameters_dic,
                    Curr_Indicator_Parameters_dic,
                    Indicator_Parameters_keys[1:])
            else:
                Indicator_analysis_helper_stop(Indicator_analysis_obj,Indicator_object,Curr_Indicator_Parameters_dic)
    def Indicator_analysis_helper_json_plot(Plot_figure_obj,
        Indicator_object,
        Link_Indicator_Parameters_dic,
        Curr_Indicator_Parameters_dic,
        Indicator_Parameters_keys):

        float_safety = 1
        if isinstance(Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][0], float):
            float_safety = float_safety_prim
        if isinstance(Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][1], float):
            float_safety = float_safety_prim
        if isinstance(Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][2], float):
            float_safety = float_safety_prim
            
        start = Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][0]*float_safety
        end = Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][1]*float_safety
        freqs = Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][2]*plot_spacing*float_safety

        for x in range(int(start),int(end),int(freqs)):
            if (float_safety == 1) and not (x == 0):
                Curr_Indicator_Parameters_dic[Indicator_Parameters_keys[0]] = int(x)
            elif x == 0:
                Curr_Indicator_Parameters_dic[Indicator_Parameters_keys[0]] = int(x)
            else:
                Curr_Indicator_Parameters_dic[Indicator_Parameters_keys[0]] = x/float_safety
            if len(Indicator_Parameters_keys) > 1:
                Indicator_analysis_helper_json_plot(Plot_figure_obj,Indicator_object,
                    Link_Indicator_Parameters_dic,
                    Curr_Indicator_Parameters_dic,
                    Indicator_Parameters_keys[1:])
            else:
                Indicator_analysis_helper_plot(Plot_figure_obj,Indicator_object,Curr_Indicator_Parameters_dic,
                    Weighted_squeeze_flo = Indicator_object['Weighted_squeeze_flo'],
                    Plot_row_int = Indicator_object['Plot_row_int'],
                    Plot_col_int = Indicator_object['Plot_col_int'], 
                    Mean_factor_bol = Indicator_object['Mean_factor_bol']
                )

    ## The Indicator analysis array
    Indicator_analysis_array = json.load(open('average_based_indicators.json'))# Opening JSON file

    ## Indicator analysis object, to analysis Indicator stock date with backtrader
    class Indicator_analysis(bt.Strategy): 
        def __init__(self):
            for indicator_object in  tqdm(Indicator_analysis_array, desc='(Process 1/3) Initialize Indicator analysis arrays (__init__) ', position=0):
                if len(list(indicator_object['Parameters_dic'].keys())) > 0 :
                    Indicator_analysis_helper_json_init(self,               #Indicator_analysis_obj
                        indicator_object,                                   #Indicator_object
                        indicator_object['Parameters_dic'],                 #Link_Indicator_Parameters_dic
                        {},                                                 #Curr_Indicator_Parameters_dic
                        list(indicator_object['Parameters_dic'].keys()))    #Indicator_Parameters_keys
                else:
                    Indicator_analysis_helper_init(self,indicator_object['Indicator_str'],{})
        def stop(self):
            for indicator_object in tqdm(Indicator_analysis_array, desc='(Process 2/3) Recording Indicator analysis arrays data (stop) ', position=0):
                if len(list(indicator_object['Parameters_dic'].keys())) > 0 :
                    Indicator_analysis_helper_json_stop(self,               #Indicator_analysis_obj
                        indicator_object,                                   #Indicator_object
                        indicator_object['Parameters_dic'],                 #Link_Indicator_Parameters_dic
                        {},                                                 #Curr_Indicator_Parameters_dic
                        list(indicator_object['Parameters_dic'].keys()))    #Indicator_Parameters_keys
                else:
                    Indicator_analysis_helper_stop(self,indicator_object,{})

    ## Using all stock
    for symbol in tqdm(symbols, desc='Running Indicator analysis "Cerebro" engine instance', position=1):
        all_data_ans = [];  colum_text = [];    file_name = symbol+'_daily_analysis'

        ## Getting the inital data
        data = engine.execute("""SELECT * FROM _daily_prices 
                                    WHERE symbol = '""" +symbol+ """'  
                                    ORDER BY symbol ASC, date ASC """)
        dfmain = pd.DataFrame(data.fetchall())
        dfmain.columns = data.keys()
        dfmain.set_index('date', inplace=True)
        analysis_data = bt.feeds.PandasData(dataname=dfmain)
        dfmain.insert(1, "date",dfmain.index[...], True) ## Adding in data of date


        ## Base Indicator analysis
        cerebro = bt.Cerebro()  # create a "Cerebro" engine instance
        cerebro.adddata(analysis_data)  # Add the data feed
        cerebro.addstrategy(Indicator_analysis)  # Add the trading strategy
        cerebro.run()#mil = 10

        ## Creating the stock plots
        fig = make_subplots(rows=1, cols=1, row_heights=[1], shared_xaxes=True)
        fig.add_trace(go.Candlestick(x=dfmain.index[...],open=dfmain['open'], high=dfmain['high'],low=dfmain['low'],close=dfmain['close'])
                ,row=1, col=1) 

        ## Adding analysised data to the plot
        for indicator_object in tqdm(Indicator_analysis_array, desc='(Process 3/3) Plotting Indicator analysis arrays data (plot) [all for:' + symbol + ']', position=0):
                if len(list(indicator_object['Parameters_dic'].keys())) > 0 :
                    Indicator_analysis_helper_json_plot(fig,                        #Plot_figure_obj                
                        indicator_object,                                   #Indicator_object
                        indicator_object['Parameters_dic'],                 #Link_Indicator_Parameters_dic
                        {},                                                 #Curr_Indicator_Parameters_dic
                        list(indicator_object['Parameters_dic'].keys()))    #Indicator_Parameters_keys
                else:
                    Indicator_analysis_helper_plot(fig,indicator_object['Indicator_str'],{})


        ## Setting axis for the layout
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            title= file_name.replace('_',' '),
            yaxis_title='Stock Price',
            xaxis_title='Date'
        )

        #custom_analysis(fig,dfmain)

        # ## Setting up paths and creating file and database
        if not os.path.exists(BASE_DIR +"/src/charts"):os.mkdir(BASE_DIR +"/src/charts")
        if not os.path.exists(BASE_DIR +"/src/charts/analysis"):os.mkdir(BASE_DIR +"/src/charts/analysis")
        if not os.path.exists(BASE_DIR +"/src/charts/analysis/html"):os.mkdir(BASE_DIR +"/src/charts/analysis/html")
        fig.write_html(BASE_DIR +"/src/charts/analysis/html/"+file_name+".html")


        if (sys.argv[3] == "create"):
            ## Removing old table if EXISTS
            engine.execute('DROP TABLE IF EXISTS average_based_indicators_analysis_table;')
            dfmain.to_sql("average_based_indicators_analysis_table", engine, if_exists='append', index=False)
            query = 'ALTER TABLE average_based_indicators_analysis_table ADD PRIMARY KEY (symbol, date);'
            engine.execute(query)
        else:
            #dfmain = dfmain.fillna(0) ## Fill in to Make sure all data is full
            dfmain.to_sql("average_based_indicators_analysis_table", engine, if_exists='append', index=False)

        del data
        del dfmain
        del analysis_data
        del cerebro
        del fig
        gc.collect()

## average_based_indicators_lines Helpers
def average_based_indicators_lines_helper_coverter(PD_indicator_object,
    Parameters_dic,
    Temp_lines_list):
    ## Calculating procedurally-generated name
    Indicator_short = PD_indicator_object['Indicator_str_short'].lower()+'_'+str("_".join(map(str, Parameters_dic.values()))).replace('.','p')
    Indicator_name = PD_indicator_object['Indicator_str'].lower()+'_'+str("_".join(map(str, Parameters_dic.values()))).replace('.','p')
    ## Setting data name as a line name for lines in backtrader
    Temp_lines_list.append([Indicator_name,Indicator_short])
def average_based_indicators_lines_json_helper_coverter(PD_indicator_object,
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
            average_based_indicators_lines_json_helper_coverter(PD_indicator_object,
                Link_Indicator_Parameters_dic,
                Curr_Indicator_Parameters_dic,
                Indicator_Parameters_keys[1:],
                Temp_lines_list)
        else:
            average_based_indicators_lines_helper_coverter(PD_indicator_object,Curr_Indicator_Parameters_dic,Temp_lines_list)
def average_based_indicators_lines_coverter(average_based_indicators):
    temp_lines = []
    for PD_indicator_object in average_based_indicators:
        if len(list(PD_indicator_object['Parameters_dic'].keys())) > 0 :
            average_based_indicators_lines_json_helper_coverter(PD_indicator_object,  #PD_indicator_object
                PD_indicator_object['Parameters_dic'],                              #Link_Indicator_Parameters_dic
                {},                                                                 #Curr_Indicator_Parameters_dic
                list(PD_indicator_object['Parameters_dic'].keys()),                 #Indicator_Parameters_keys
                temp_lines)                                                         #Temp_lines_list    
        else:
            average_based_indicators_lines_helper_coverter(PD_indicator_object['Indicator_str'],{},temp_lines)
    return temp_lines

def average_based_indicators_genes_table():
    ### Create can only do one stock at a time
    
    ##START
    print("Starting average_based_indicators gene table creation ... this could take a while")

    ## Getting symbolsto chart
    symbols = sys.argv[2].split(",")

    #if all getting all from database
    if symbols[0] == "all":
        data = engine.execute("SELECT symbol FROM _daily_prices GROUP BY symbol")
        symbols = [item['symbol'] for item in data.fetchall()]
    print("Stocks for analysis :",symbols)

    start = '2004-12-31 00:00:00'
    end = '2021-07-08 00:00:00'

    ## The Indicator analysis array
    average_based_indicators = json.load(open('average_based_indicators.json'))# Opening JSON file
    average_based_indicators_lines = average_based_indicators_lines_coverter(average_based_indicators)
    crossover_groups =list(permutations(average_based_indicators_lines, 2))

    create_tables = False
    if(sys.argv[3] == "create"):
        create_tables = True
    if(create_tables):
        for lines in tqdm(average_based_indicators_lines,desc=('Creating tables and line, please wait a sec'), position=0):
            engine.execute('DROP TABLE IF EXISTS average_based_indicators_genes_table_%s;'% (str(lines[1])))## Firing query statment
            query = """
                    CREATE TABLE average_based_indicators_genes_table_%s AS
                        SELECT date, symbol
                        FROM average_based_indicators_analysis_table;
                    """% (str(lines[1]))
            engine.execute(query) ## Firing query statment
            engine.execute('ALTER TABLE average_based_indicators_genes_table_%s ADD PRIMARY KEY (symbol, date);'% (str(lines[1])))## Firing query statment
        create_tables = False

    def one_process(average_base):
        ## Adding in lines of the dates and the symbol
        for symbol in symbols:
            query = ("""
                SELECT symbol,date FROM public.average_based_indicators_analysis_table
                    WHERE symbol = '%s'
                    ORDER BY symbol ASC, date ASC
            """ %
            (symbol))

            date_symbol_data = engine.execute(query)
            date_symbol_data_df = pd.DataFrame(date_symbol_data.fetchall())
            date_symbol_data_df.columns = date_symbol_data.keys()
            date_symbol_data_df.to_sql(('average_based_indicators_genes_table_%s'%(average_base[1])),engine, if_exists='append', index=False) 

            del date_symbol_data
            gc.collect()

    def for_process(crossover_group):
        gene_name = ("crossover_%s"%("_x_".join([crossover_group[0][1],crossover_group[1][1]])))
        for symbol in symbols:

            if(sys.argv[3] == "create"):
                query = ('ALTER TABLE IF EXISTS ONLY %s ADD COLUMN %s %s' % (("average_based_indicators_genes_table_%s" % crossover_group[0][1]), gene_name , "DOUBLE PRECISION"))
                engine.execute(query) ## Firing query statment
            
        
            ## Updateing colume information
            query = ("""
                    BEGIN;
                        SET LOCAL lock_timeout = '2s';
                        UPDATE average_based_indicators_genes_table_%s
                        SET %s =
                            (SELECT (%s-%s) FROM average_based_indicators_analysis_table
                            WHERE average_based_indicators_analysis_table.symbol = '%s'
                            AND average_based_indicators_analysis_table.date = average_based_indicators_genes_table_%s.date)
                        WHERE average_based_indicators_genes_table_%s.symbol = '%s';
                    COMMIT;
                """ % #(gene_name,crossover_group[0][1],crossover_group[0][1],symbol,crossover_group[0][1],crossover_group[0][1],
                (crossover_group[0][1],gene_name,
                    crossover_group[0][0],crossover_group[1][0],symbol,crossover_group[0][1],
                crossover_group[0][1],symbol))

            engine.execute(query) ## Firing query statment
            del query
            del symbol
            gc.collect()

        del gene_name
        del crossover_group
        gc.collect()

    base = len(average_based_indicators_lines)
    mass = len(crossover_groups)

    # for average_base in tqdm(average_based_indicators_lines,desc=('Initializing gene analysis table'), position=0):
    #     one_process(average_base)
    #     del average_base
    #     gc.collect()


    for crossover_group in tqdm(crossover_groups,desc=('Creating CrossOver Genes for (Full Load is %s)'%(str(mass/base))), position=0):
        for_process(crossover_group)
        del crossover_group
        gc.collect()
        #time.sleep(1)


#Running sim to test fitness
def test_fitness(test_organisms=[],stock_index=0,testing=True,test_symbols_stock_data=[],test_symbols_gene_data=[],crossover_groups=[]):
    gene_spacer = gene_spacer_main_MT
    test_organisms = test_organisms['dna']
    #print(test_organisms)
    # list of parameters which are configurable for the strategy
    class genetic_algorithm_organism(bt.Strategy):
        def __init__(self):
            self.counter = 0

        def next(self):

            buy = 0; not_buy = 0; sell = 0; not_sell = 0; total = len(crossover_groups)

            if self.counter > 106:
                for idx, crossover in enumerate(crossover_groups):
                    gene_name = ("crossover_%s"%("_x_".join([crossover[0][1],crossover[1][1]])))
                    
                    # last5day_crossover = []
                    # for i in range(6):
                    #     data_CO1 = test_symbols_gene_data[stock_index][gene_name][self.counter-i-1]
                    #     data_CO2 = test_symbols_gene_data[stock_index][gene_name][self.counter-i]
                    #     ans = 0
                    #     if not (np.sign(data_CO1) == np.sign(data_CO2)) : ans = 1
                    #     last5day_crossover.append(ans)

                    # basic crossover
                    data_CO1 = test_symbols_gene_data[stock_index][gene_name][self.counter-1]
                    data_CO2 = test_symbols_gene_data[stock_index][gene_name][self.counter]
                    ans = 0

                    if not (np.sign(data_CO1) == np.sign(data_CO2)) : ans = 1
                    buy = (test_organisms[(idx*gene_spacer)+0] * ans) + buy
                    sell = (test_organisms[(idx*gene_spacer)+1] * ans) + sell

                    ## FROMR 4 PART vvv , SWITCHING TO 2 PARTS ^^^^

                    # if not (np.sign(data_CO1) == np.sign(data_CO2)) : ans = 1
                    # buy = (test_organisms[(idx*gene_spacer)+0] * ans) + buy
                    # not_buy = (test_organisms[(idx*gene_spacer)+1] * ans) + not_buy
                    # sell = (test_organisms[(idx*gene_spacer)+2] * ans) + sell
                    # not_sell = (test_organisms[(idx*gene_spacer)+3] * ans) + not_sell

                    # # Last 5 days crossover
                    # ans = 0
                    # if 1 in last5day_crossover[1:]: ans = 1
                    # buy = (test_organisms[(idx*gene_spacer)+4] * ans) + buy
                    # not_buy = (test_organisms[(idx*gene_spacer)+5] * ans) + not_buy
                    # sell = (test_organisms[(idx*gene_spacer)+6] * ans) + sell
                    # not_sell = (test_organisms[(idx*gene_spacer)+7] * ans) + not_sell

                

                ## My safey net
                perce = (self.broker.getvalue()/700) * 29
                cerebro.addsizer(bt.sizers.PercentSizer, percents=perce)

                if testing :
                    print('\t','counter:',self.counter,'buy:',buy/total,'not_buy:',not_buy/total,'sell:',sell/total,'not_sell:',not_sell/total,'total:',total)
                    print('\t\t','buy_TH:',test_organisms[-1],'sell_TH:',test_organisms[-2],'should not be th:',test_organisms[-3],'should not be th:',test_organisms[-4])
                    print('\t\t','Holding:',self.position.size,'Value:',self.broker.getvalue(),'next percents',perce)


                if not self.position:  # not in the market
                    if (buy/total) > test_organisms[-1]:  
                        self.buy()  # enter long
                        if testing :print("\t\t\t Buy")
                    else:
                        if testing :print("\t\t\t could Buy but did not")
                else:
                    if (sell/total) > test_organisms[-2]: 
                        self.close()
                        if testing :print("\t\t\t Sell")
                    else:
                        if testing :print("\t\t\t could Sell but did not")
                       

                ## FROMR 4 PART vvv , SWITCHING TO 2 PARTS ^^^^
                # if not self.position:  # not in the market
                #     if not (not_buy/total) > test_organisms[-2]:
                        
                #         if (buy/total) > test_organisms[-1]:  
                #             self.buy()  # enter long
                #             if testing :print("\t\t\t Buy") 
                #         else:
                #             if testing :print("\t\t\t could Buy")
                #     else:
                #         if testing :print("\t\t\t Not Buy")
                # else:
                #     if not (not_sell/total) > test_organisms[-4]:
                #         if (sell/total) > test_organisms[-3]: 
                #             self.close()
                #             if testing :print("\t\t\t Sell")
                #         else:
                #             if testing :print("\t\t\t could Sell")
                #     else:
                #         if testing :print("\t\t\t Not Sell")

            self.counter = self.counter + 1

    #print("test_fitness Cerebro start")
    cerebro = bt.Cerebro()  # create a "Cerebro" engine instance
    bt_data = bt.feeds.PandasData(dataname=test_symbols_stock_data[stock_index])
    cerebro.broker.set_cash(1000)
    cerebro.broker.setcommission(commission=0.10)
    cerebro.adddata(bt_data)  # Add the data feed
    cerebro.addstrategy(genetic_algorithm_organism)  # Add the trading strategy
    cerebro.addsizer(bt.sizers.PercentSizer, percents=20)  # default sizer for strategies
    cerebro.run() # run it all
    #print("test_fitness Cerebro end")
    # cerebro.plot() # and plot it with a single command

    #print('Final Portfolio Value : %0.2f' % cerebro.broker.getvalue())

    return cerebro.broker.getvalue()

def average_based_indicators_genetic_algorithm():
    """
    an events : Meaning grouping of species before any time of intervention for me to make a the algorithm 
        better or (hopefully not) worsts 
    """
    id_event = 2 # What numder event is this
    nm_event = 'Main_Run_2part' # What is that name of this event

    """
    a species : Meaning a group of "Organisms" underging simulated natural selection, in each run of 
        the genetic algorithm, a species will have severals generation each containing a populations 
        of "Organisms" 
    """
    num_species = 3 # 20 # Number of Species per event

    """
    a generation : Meaning a group of "Organisms" in a populations that are the result of one 
        simulated natural selection, a generation will have a populations, and this population 
        will have it's fitness tested to see whichs 2 to 4 to be used to make the next generation's 
        populations of "Organisms". The number of generations tells how make times simulated 
        natural selection happen.
    """
    num_generation = 40 #28 = 2 days # 50 # Number of generations per species
    # num_reproducers = 2 # Number "Organisms" with the top fitness that are breed ##Not in use anymore
    num_mutation = 0.15 #20%  # how much high% or low%, could a gene mutation during breeding


    """
    an organism : A member of a populations, meaning a group of genes and weights the determin how much the
    storge that gene  in a populations that are the result of one 
        simulated natural selection, a generation will have a populations, and this population 
        will have it's fitness tested to see whichs 2 to 4 to be used to make the next generation's 
        populations of "Organisms". The number of generations tells how make times simulated 
        natural selection happen.
    """
    num_organisms = 16 # 20 # Number of organisms per population
    
     
    ## The Indicator analysis array
    average_based_indicators = json.load(open('average_based_indicators.json'))# Opening JSON file
    average_based_indicators_lines = average_based_indicators_lines_coverter(average_based_indicators)
    crossover_groups =list(permutations(average_based_indicators_lines, 2))

    #data = engine.execute("SELECT symbol FROM average_based_indicators_genes_table_%s GROUP BY symbol"%average_based_indicators_lines[0][1])
    #test_symbols = [item['symbol'] for item in data.fetchall()]
    #test_symbols = ['AAU', 'AMZN', 'JYNT', 'MBI', 'OCGN', 'TWTR']
    test_symbols = ['AAL', 'AAU', 'AMZN', 'BB', 'BSAC', 'CDAY','CELC','DRUA','EYEN','GOLD','JYNT','MBI','NSS','OCGN','SD','SPH','TWTR']
    #test_symbols = ['JYNT','MBI','SD']

    #test_symbols = ['JYNT']
    # test_symbols_gene_data = []
    # test_symbols_stock_data = []
    start = '2004-12-31 00:00:00'
    end = '2021-07-08 00:00:00'

    def get_stock_data_GA(test_symbols):
        test_symbols_stock_data = [] 
        for test_symbol in tqdm(test_symbols,desc='Loading stock data from database', position=0):
            query = """
            SELECT * FROM average_based_indicators_analysis_table 
                WHERE symbol = '%s' 
                ORDER BY symbol ASC, date ASC
            """% (test_symbol)

            raw_data = engine.execute(query)
            raw_stock_data = pd.DataFrame(raw_data.fetchall())
            raw_stock_data.columns = raw_data.keys()
            raw_stock_data = raw_stock_data.set_index(raw_stock_data['date'])
            test_symbols_stock_data.append(raw_stock_data)
        return test_symbols_stock_data

    def get_gene_data__GA(test_symbols):
        test_symbols_gene_data = []
        for test_symbol in test_symbols:
            df = pd.DataFrame([])
            for half_genes in tqdm(average_based_indicators_lines,desc='Loading Gene data from database for %s'%test_symbol, position=0):
                query = """
                SELECT * FROM public.average_based_indicators_genes_table_%s
                    WHERE symbol = '%s'
                    ORDER BY date ASC, symbol ASC
                """ % (half_genes[1],test_symbol)

                raw_data = engine.execute(query)
                df_sub = pd.DataFrame(raw_data.fetchall())
                df_sub.columns = raw_data.keys()
                df_sub = df_sub.set_index(df_sub['date'])
                df = pd.concat([df, df_sub], axis=1)

                del raw_data
                del df_sub
                gc.collect()
            
            df = df.groupby(level=0).sum()
            test_symbols_gene_data.append(df)
            del df
            del test_symbol
            gc.collect()
        gc.collect()

        return test_symbols_gene_data
    
    # test_symbols_stock_data = get_stock_data_GA(test_symbols)
    # test_symbols_gene_data = get_gene_data__GA(test_symbols)

    gene_spacer_main = gene_spacer_main_MT
    

    # test_organisms = make_organisms(len(crossover_groups))

    # stock_index =  0

    # print(test_organisms[-1],test_organisms[-2])

    # for count in range(4):
    #     test_organisms[-1 - count] = 0.05
    
    #print('\t\t','buy_TH:',test_organisms[-1],'not_buy_TH:',test_organisms[-2],'sell_TH:',test_organisms[-3],'not_sell_TH:',test_organisms[-4])
    # test_organisms[-1] = 1/8930
    # test_organisms[-2] = 1/8930
    # print(test_organisms[-1],test_organisms[-2])

    ## Makes single raw set of DNA
    def make_initial_organism_dna(len_int):

        gene_spacer = gene_spacer_main
        num_list = []
        for count in range((len_int*gene_spacer)+gene_spacer):
            #num_list.append((random.randint(-400, 400)/100))
            #num_list.append(0.5)
            num_list.append((random.randint(150, 850)/1000))

        num_list[-1] = 0.020 # BUY
        num_list[-2] = 0.020 # SELL

        ## FROMR 4 PART vvv , SWITCHING TO 2 PARTS ^^^^
        # gene_spacer = gene_spacer_main
        # num_list = []
        # for count in range((len_int*gene_spacer)+gene_spacer):
        #     #num_list.append((random.randint(-400, 400)/100))
        #     #num_list.append(0.5)
        #     num_list.append((random.randint(350, 650)/1000))

        # num_list[-1] = 0.010
        # num_list[-2] = 0.015
        # num_list[-3] = 0.010
        # num_list[-4] = 0.015

        return num_list

    ## Create inital generation 
    def creating_initial_generation(num_organisms,id_generation,id_species):
        initial_generation = []
        for id_organism in tqdm(range(1,num_organisms+1),desc=('Creating initial generation'), position=0):
            dna_organism = make_initial_organism_dna(len(crossover_groups))
            organisms = {
                "id_organism": id_organism,
                "id_event": "%s_%s"%(str(id_event),nm_event),
                "id_generation":id_generation,
                "id_species":id_species,
                "dna": dna_organism,
                "birth": pd.to_datetime('now'),## Add Updata timestamp
                "origincodes":"CIG",
                "parents_1": "CIG",
                "parents_2": "CIG"
            }
            initial_generation.append(organisms)

        return initial_generation

    ## Create inital generation 
    def test_generation(curr_generation_organisms,stock_index,create_tables_2,testing):
        curr_generation_fitness = []

        ## For testing
        # # curr_generation_fitness = [17.778084103130116, 20.964289019384296, 20.956819963959955]
        # # num_list = []
        # for count in range(len(curr_generation_organisms)):
        #     #num_list.append((random.randint(-400, 400)/100))
        #     #num_list.append(0.5)
        #     curr_generation_fitness.append((random.randint(350, 650)/100))

        test_symbols_stock_data = get_stock_data_GA([test_symbols[stock_index]])
        test_symbols_gene_data = get_gene_data__GA([test_symbols[stock_index]])

        
        pool_partial = partial(test_fitness, stock_index=0, testing=testing, 
            test_symbols_stock_data=test_symbols_stock_data, test_symbols_gene_data=test_symbols_gene_data, crossover_groups=crossover_groups)

        with Pool(3) as p:
            curr_generation_fitness = list(tqdm(p.imap(pool_partial, curr_generation_organisms), 
                total=len(curr_generation_organisms),
                desc=('Testing (the fitness) for generation %s using %s '%(curr_generation_organisms[0]['id_generation'] , test_symbols[stock_index])))
                )

        # ## for testing as well
        # for each in tqdm(curr_generation_organisms, # x = each orgnsima
        #     desc=('Testing (the fitness) for generation %s using %s '%(curr_generation_organisms[0]['id_generation'] , test_symbols[stock_index])), # in %s'%str(curr_generation_topps[0]['id_generation'])), 
        #     position=0,
        #     total=len(curr_generation_organisms)): 
        #     test_fitness(test_organisms=each,stock_index=0,testing=testing,test_symbols_stock_data=test_symbols_stock_data,
        #         test_symbols_gene_data=test_symbols_gene_data,crossover_groups=crossover_groups)
    

        print(('Generation %s fitness with stock %s >\t>'%(curr_generation_organisms[0]['id_generation'] , test_symbols[stock_index])),curr_generation_fitness)
      
        #curr_generation_fitness.append(test_fitness(organisms['dna'],stock_index,testing))

        return curr_generation_fitness

    ## Create table if not created already
    def create_save_generation_data_table_lines(test_symbols_gene_data):  
        lines = []
        lines.append('fitness')
        lines.append('id_event')
        lines.append('id_organism')
        lines.append('id_generation')
        lines.append('id_species')
        lines.append('birth')
        lines.append('id_origincodes')
        lines.append('parents_1')
        lines.append('parents_2')
        lines.append('organisms_name')

        # if not gene_spacer_main == 4:
        #     raise ValueError('create_save_generation_data_table_lines is not syced to gene_spacer_main')
        # for test_symbols_gene in tqdm(test_symbols_gene_data[0],
        #     desc=('Creating Save data table lines'), 
        #     position=0,
        #     total=len(test_symbols_gene_data[0])):
        #         lines.append("gene_%s_1d_BBuy"%test_symbols_gene)
        #         lines.append("gene_%s_1d_NBuy"%test_symbols_gene)
        #         lines.append("gene_%s_1d_SSel"%test_symbols_gene)
        #         lines.append("gene_%s_1d_NSel"%test_symbols_gene)
        #         # if gene_spacer_main == 4 than there should be 4 append

        lines.append("gene_main_buy_TH")
        lines.append("gene_main_not_buy_TH")
        lines.append("gene_main_sell_TH")
        lines.append("gene_main_not_sell_TH")

        return lines
    ## data_table_lines = create_save_generation_data_table_lines(test_symbols_gene_data)

    ## Save data to the database
    def save_generation_data(curr_generation_fitness,curr_generation_organisms,stock_index,create_tables):  
        organisms_intel = [] 

        if not os.path.exists(BASE_DIR +"/src/genetic_algorithm"):os.mkdir(BASE_DIR +"/src/genetic_algorithm")
        if not os.path.exists(BASE_DIR +"/src/genetic_algorithm/gene_codes"):os.mkdir(BASE_DIR +"/src/genetic_algorithm/gene_codes") 

        for idx,organisms in tqdm(enumerate(curr_generation_organisms),
            desc=('Save generation date for %s'%str(curr_generation_organisms[0]['id_generation'])), 
            position=0,
            total=len(curr_generation_organisms)):
            organisms_intel_sub = [curr_generation_fitness[idx]]
            organisms_intel_sub.extend([organisms["id_event"],int(organisms["id_organism"]),int(organisms["id_generation"])])
            organisms_intel_sub.extend([int(organisms["id_species"]),organisms["birth"],organisms["origincodes"]])
            organisms_intel_sub.extend([organisms["parents_1"],organisms["parents_2"]])

            #organisms name
            organisms_name = "organism_e%s_s%s_g%s_o%s"%(str(organisms['id_event']),str(organisms['id_species'])
                ,str(organisms['id_generation']),str(organisms['id_organism']))
            organisms_intel_sub.extend([organisms_name])
            organisms['organisms_name'] = organisms_name
            #Saveing Raw DNA
            Raw_DNA = pd.DataFrame(organisms["dna"][0:-5])
            Raw_DNA.reset_index(drop=True)
            #print(Raw_DNA) 
            Raw_DNA.to_csv(("%s/src/genetic_algorithm/gene_codes/%s.csv"%(BASE_DIR,organisms_name)))

            #print('gene',organisms["dna"])
            #print('gene main TH, should be 4 of them all under 0.06',organisms["dna"][-4:])
            organisms_intel_sub.extend(organisms["dna"][-4:])

            organisms_intel.append(organisms_intel_sub)
        
        print('Finishing up save process for generation %s'%str(curr_generation_organisms[0]['id_generation']))

        if create_tables == True:
            engine.execute('DROP TABLE IF EXISTS _generation_data;')
        lines = create_save_generation_data_table_lines([])#(test_symbols_gene_data)
        result = pd.DataFrame(organisms_intel, columns=lines)

        #print(result.columns[-5])
        cols = list(result)
        cols.insert(0, cols.pop(-5))
        result = result.loc[:, cols]
        result.set_index(result.columns[0])

        # print(result)
        
        if create_tables == True:
            result.to_sql('_generation_data',engine, if_exists='append')
            query = 'ALTER TABLE _generation_data ADD PRIMARY KEY (organisms_name,id_species,id_generation,id_organism,birth);'
            engine.execute(query)
            create_tables = False
        else:
            result.to_sql('_generation_data',engine, if_exists='append') 




        #     for test_symbols_gene in test_symbols_gene_data[stock_index]:
               
        #         # df[(test_symbols_gene+"5d_BBuy")] = organisms['dna'][count+4]
        #         # df[(test_symbols_gene+"5d_NBuy")] = organisms['dna'][count+5]
        #         # df[(test_symbols_gene+"5d_SSel")] = organisms['dna'][count+6]
        #         # df[(test_symbols_gene+"5d_NSel")] = organisms['dna'][count+7]
        #         count = count + gene_spacer_main
            
        #     df["main_buy_TH"] = organisms['dna'][-1]
        #     df["main_not_buy_TH"] = organisms['dna'][-2]
        #     df["main_sell_TH"] = organisms['dna'][-3]
        #     df["main_not_sell_TH"] = organisms['dna'][-4]
        #     df['fitness'] = curr_generation_fitness[idx]

        #     result = pd.concat([result,df])
        #     del organisms["dna"]

        # result = pd.concat([result,pd.DataFrame(organisms)])
        # print(result)
        # result.to_sql('_generation_data',engine, if_exists='append')
            
    def selection_operator(curr_generation,curr_generation_fitness):
        top = sorted(range(len(curr_generation_fitness)), key=lambda i: curr_generation_fitness[i])[-2:]
        print('Selecting the best of the generation',curr_generation[top[1]]['id_generation']
            ,', there scores :',curr_generation_fitness[top[0]],curr_generation_fitness[top[1]])
        return curr_generation[top[0]],curr_generation[top[1]]

    def crossover_operator(curr_generation_topps):

        #hard coded to save time
        # def crossover(p1,p2):
        cut_len_on = gene_spacer_main

        all_parts = [[None for _ in range(2)] for _ in range(cut_len_on)]
        all_parts_text = [[None for _ in range(2)] for _ in range(cut_len_on)]
        #print(all_parts,all_parts_text)
        for a in  range(len(curr_generation_topps)):
            for i in range(cut_len_on):
                part = np.array(curr_generation_topps[a]['dna'][i::cut_len_on])
                all_parts[i][a] = part.tolist()
                all_parts_text[i][a] = ('P%sSC%s'%((a+1),(i+1)))

        # all_parts = list(product(*all_parts))
        # all_parts_text = list(product(*all_parts_text))

        #pprint.pprint(all_parts_text)
        # pprint.pprint(crossover_text)

        ## a = [[1,2,3],[4,5,6],[7,8,9,10]]


            # p1B = np.array[curr_generation_topps[0]['dna'][1::cut_len_on]]
            # p1C = np.array[curr_generation_topps[0]['dna'][2::cut_len_on]]
            # p1D = np.array[curr_generation_topps[0]['dna'][3::cut_len_on]]

            # p2A = np.array[curr_generation_topps[1]['dna'][0::cut_len_on]]
            # p2B = np.array[curr_generation_topps[1]['dna'][1::cut_len_on]]
            # p2C = np.array[curr_generation_topps[1]['dna'][2::cut_len_on]]
            # p2D = np.array[curr_generation_topps[1]['dna'][3::cut_len_on]]



            # c1 = np.append(p1A,p2B,p2C,p2D)
            # c2 = np.append(p1A,p2B,p2C,p2D)
            # c3 = 
            # c4 = 




        """
        I could not get theway i wanted to do it to work so,ill have to do it the generic single point 
        way in order to meet the deadline

        """

        # def chunkIt(seq, num):
        #     avg = len(seq) / float(num)
        #     out = []
        #     last = 0.0

        #     while last < len(seq):
        #         out.append(seq[int(last):int(last + avg)])
        #         last += avg

        #     return out

        # ## running out of time so will have to hard code the crossover_operator in the Mutation Operator
        # p1 = curr_generation_topps[0]['dna']
        # p2 = curr_generation_topps[1]['dna']




        # # p1c1 = curr_generation_topps[0]['dna'][]
        # # p1c2 = curr_generation_topps[0]['dna']
        # # p1c3 = curr_generation_topps[0]['dna']
        # # p1c4 = curr_generation_topps[0]['dna']

        # # p2c1 = curr_generation_topps[1]['dna']
        # # p2c2 = curr_generation_topps[1]['dna']
        # # p2c3 = curr_generation_topps[1]['dna']
        # # p2c4 = curr_generation_topps[1]['dna']

        # children_list = []

        # cut_len_on = gene_spacer_main
        # all_parts = []
        # all_parts_text = []

        # for i in tqdm(range(cut_len_on),
        #     desc=('Crossing over (Breeding) most fit organisms in %s'%str(curr_generation_topps[0]['id_generation'])), 
        #     position=0):
        #     all_parts.append(curr_generation_topps[0]['dna'][(0+i)::cut_len_on])
        #     all_parts_text.append('P1SC%s'%(i))
        #     all_parts.append(curr_generation_topps[1]['dna'][(0+i)::cut_len_on])
        #     all_parts_text.append('P2SC%s'%(i))


        # children_list = list(combinations_with_replacement(all_parts, cut_len_on))
        # #children_list = children_list[:int(len(children_list)/4)]
        # children_list_text = list(combinations_with_replacement(all_parts_text, cut_len_on))
        # #children_list_text = children_list_text[:int(len(children_list)/4)]

        # print(len(all_parts_text))
        #print(children_list)

        # for children in all_parts_text:
        #     print(children)


        # cut_len = len(curr_generation_topps[0]['dna'])/4
        # all_part = pprint.pprint(list(chunks(curr_generation_topps[0]['dna'], cut_len)))
        # all_part.extend(pprint.pprint(list(chunks(curr_generation_topps[1]['dna'], cut_len))))

        # ## Running out of time, so will have to hard code in, sorry 
        # c1 = p1[int(len(p1)/2):] + p2[int(len(p1)/2):] ## Child 1 ##origincodes P1GSBP2SB
        # c2 = p1[:int(len(p1)/2)] + p2[int(len(p1)/2):] ## Child 1 ##origincodes P1GSBP2SB

        # c1_organisms = {
        #         "id_organism": 1,
        #         "id_event": "%s_%s"%(str(id_event),nm_event),
        #         "id_generation":(curr_generation_topps[0]['id_generation']+1),
        #         "id_species":curr_generation_topps[0]['id_species'],
        #         "dna": c1,
        #         "birth": pd.to_datetime('now'),## Add Updata timestamp
        #         "origincodes":"CIG"
        #     }

        # print(c1_organisms)
        del curr_generation_topps
        return [all_parts,all_parts_text]

    def mutation_operator(curr_children):
        # running out of time fast, this is not the best code

        # # mutation_operator
        # all_dna = curr_children[0].copy()
        # all_dna_text = curr_children[1].copy()
        all_parts_2 = [[None for _ in range(len(curr_children[0][0]))] for _ in range(len(curr_children[0]))]
        all_parts_text_2 = [[None for _ in range(len(curr_children[0][0]))] for _ in range(len(curr_children[0]))]

        #pprint.pprint(all_parts_text_2)

        for x in tqdm(range(len(curr_children[0])),
            desc=('Prepareing mutations to make the new children'), # in %s'%str(curr_generation_topps[0]['id_generation'])), 
            position=0):  
            for y in range(len(curr_children[0][0])):
                #np.insert(all_dna_text[x],int(y*2),curr_children[1][x][y].replace('S','M'))
                all_parts_text_2[x][y] = curr_children[1][x][y].replace('S','M')
                new_dna = []

                for dna in curr_children[0][x][y]:
                   new_dna.append(random.randint(int(dna*10000*(1-num_mutation)),int(dna*10000*(1+num_mutation)))/10000)
                all_parts_2[x][y] = new_dna


                ## a possable change
                # all_parts_text_2[x][y] = curr_children[1][x][y].replace('S','M')
                # new_dna = []
                # th = curr_children[0][x][y][-1]
                # mean = sum(curr_children[0][x][y]) / len(curr_children[0][x][y])
                # for dna in curr_children[0][x][y]:
                #    new_dna.append(random.randint(int(mean*10000*(1-num_mutation)),int(mean*10000*(1+num_mutation)))/10000)
                # all_parts_2[x][y] = new_dna
                # all_parts_2[x][y][-1] = random.randint(int(th*10000*(1-num_mutation)),int(th*10000*(1+num_mutation)))/10000
 

        # del curr_children
        all_dna = []
        all_dna_text = []
        for x in range(len(all_parts_2)):
            ans_data_sub = []
            ans_data_text_sub = []
            for y in range(len(all_parts_2[x])):
                ## date 
                ans_data_sub.append(all_parts_2[x][y])
                ans_data_sub.append(curr_children[0][x][y])
                ## text 
                ans_data_text_sub.append(all_parts_text_2[x][y])
                ans_data_text_sub.append(curr_children[1][x][y])
            all_dna.append(ans_data_sub) 
            all_dna_text.append(ans_data_text_sub)

        return [all_dna,all_dna_text] #[all_parts,all_parts_text]
        
    def assembling_operator(curr_children,old_teacher):
        all_dna = list(product(*curr_children[0]))
        all_dna_text = list(product(*curr_children[1]))


        #all_dna[0] organmsin
        # print(all_dna[0][0])#This is how tha parts look
        # print(all_dna[0][0][-1]) # thats a gene
        # print(all_dna_text[0][0])
        # pprint.pprint(all_dna_text)
        all_dna_assembling = []
        all_dna_text_assembling = []
        

        for x in tqdm(range(len(all_dna)), # x = each orgnsima
            desc=('creating (assembling 1) generation %s'%(old_teacher[0]['id_generation']+1)), # in %s'%str(curr_generation_topps[0]['id_generation'])), 
            position=0,
            total=len(all_dna)): 


            #all_dna_sub = list(zip(*all_dna[x]))
            all_dna_sub = list(chain.from_iterable(list(zip(*all_dna[x]))))


            # all_dna_sub = [None for _ in range(len(all_dna[0][0])*gene_spacer_main)]
            all_dna_sub_text = ''

            # for y in range(len(all_dna[x])): # y = eacg part
            #     for a in range(len(all_dna[x][y])): # a = eacg gene
            #         print('\t\t\t',(all_dna_sub[(y+1)*(1+a)-2] == None))
            #         all_dna_sub[(y+1)*(1+a)-2] = all_dna[x][y][a]
            #         print('\t\t',x,y,a,(y+1)*(1+a)-2,all_dna_sub[(y+1)*(1+a)-2])



                






            # all_dna_sub = [] # [None for _ in range(len(all_dna_text)*gene_spacer_main)]
            # all_dna_sub_text = ''

            # np_temp_holder = [] 
            # for y in range(len(all_dna[x])):
            #     np_temp_holder.append(np.array(all_dna[x][y]))

            # for y in range(1,len(np_temp_holder)):
            #     np_temp_holder[y] = np.insert(np_temp_holder[y], np.arange(len(np_temp_holder[y-1])), np_temp_holder[y-1])




            # np.insert(B, np.arange(len(A)), A)

            # #np.insert(B, np.arange(len(A)), A)

            # for y in range(len(all_dna[x])):


            #     #all_dna_sub[x*gene_spacer_main] = 
            #     all_dna_sub.insert(y, all_dna[x][y])

            #     #list(map(lambda(i, j): i + j, zip(all_dna_sub, test_list2)))
            #     #all_dna_sub.extend(gene)

            for gene in all_dna_text[x]:
                all_dna_sub_text= all_dna_sub_text + gene

            # print('sub',all_dna_sub[-4],all_dna_sub[-3],all_dna_sub[-2],all_dna_sub[-1])
            # print('text',all_dna_sub_text)
                #all_dna_sub_text = all_dna_sub_text + gene

            # for y in range(len(all_dna[x])):
            #     print(all_dna_sub_text)
            #     all_dna_sub.extend(all_dna[x][y])
            #     all_dna_sub_text.extend(all_dna_text[x][y])

            
            all_dna_assembling.append(all_dna_sub)
            all_dna_text_assembling.append(all_dna_sub_text)

        # print(all_dna_sub)
        # #print('all', all_dna_text_assembling)
        #print(all_dna_assembling[0])
        

        # def chunks(lst, n):
        #     """Yield successive n-sized chunks from lst."""
        #     for i in range(0, len(lst), n):
        #         yield lst[i:i + n]

        # part_groups = list(chunks(curr_children[0], gene_spacer_main))
        # part_groups_text = list(chunks(curr_children[1], gene_spacer_main))
        # #pprint.pprint(part_groups)
        # ans=[]
        # ans_text=[]

        # #hard code fix, because the deadline is coming fast
        # for x in tqdm(range(gene_spacer_main),
        #     desc=('creating (assembling 1) generation %s'%(old_teacher['id_generation']+1)), # in %s'%str(curr_generation_topps[0]['id_generation'])), 
        #     position=0,
        #     total=gene_spacer_main): 

        #     for y in range(gene_spacer_main):
        #         # for z in range(gene_spacer_main):
        #         #     for i in range(gene_spacer_main):
        #         #ans_text.append('%s_%s_%s_%s'%(part_groups_text[0][x],part_groups_text[1][y],part_groups_text[2][z],part_groups_text[3][i]))
        #         ans_text.append('%s_%s'%(part_groups_text[x][y],part_groups_text[y][x])) #,part_groups_text[2][z],part_groups_text[3][i]))
        #         ans.append(part_groups[x][y]+part_groups[y][x]) #+part_groups[2][z]+part_groups[3][i])

        # pprint.pprint(ans_text)

        curr_gen = []
        for x in tqdm(range(len(all_dna_assembling)),
            desc=('birthing (assembling 2) generation %s'%(old_teacher[0]['id_generation']+1)), # in %s'%str(curr_generation_topps[0]['id_generation'])), 
            position=0,
            total=len(all_dna_assembling)):

            #
            #if all_dna_assembling[x][-1] > all_dna_assembling[x][-2]: # h
        #     df["main_not_buy_TH"] = organisms['dna'][-2]
        #     df["main_sell_TH"] = organisms['dna'][-3]
        #     df["main_not_sell_TH"] = organisms['dna'][-4]

            curr_gen.append({
                "id_organism": 1+x,
                "id_event": old_teacher[0]['id_event'],
                "id_generation": (old_teacher[0]['id_generation']+1),
                "id_species": old_teacher[0]['id_species'],
                "dna": all_dna_assembling[x],
                "birth": pd.to_datetime('now'),## Add Updata timestamp
                "origincodes": all_dna_text_assembling[x],
                "parents_1": old_teacher[0]['organisms_name'],
                "parents_2": old_teacher[1]['organisms_name']
            })


        return curr_gen 

    create_tables_2 = False

    if sys.argv[3] == 'create':
        create_tables_2 = True


    # curr_generation = creating_initial_generation(9,1,1) # (num_organisms,id_generation,id_species)
    # curr_generation_fitness = test_generation(curr_generation,0,create_tables_2,False)
    # print(curr_generation_fitness)
    # curr_generation_fitness_t = [1,1,1,1,23,45,123,12]


    # curr_selection = selection_operator(curr_generation,curr_generation_fitness_t)
    # children_list_children_list_text = crossover_operator(curr_selection)
    # children_list_children_list_text = mutation_operator(children_list_children_list_text)
    # curr_generation = assembling_operator(children_list_children_list_text,curr_generation[0])

    start_main = time.time()
    print('-------------------------------------------------------------------------------- Whole Job Starting | with ',(num_species-1),'species')
    for id_species in range(1,num_species):
        print('----------------------------------------------------------------------------------------------- Starting | Species:',id_species) 
        curr_generation = creating_initial_generation(num_organisms,1,1) ## Randomly initialize populations
        for id_population in range(1,num_generation):  
            start_pop = time.time()  
            print('--------------------- Testing Startng') 
            curr_generation_fitness = []
            list_of_random_items = random.sample(range(len(test_symbols)), 3)
            for i in range(3):
                #stock_index = random.choice(range(len(test_symbols)))
                stock_index = list_of_random_items[i]
                curr_generation_fitness.append(test_generation(curr_generation,stock_index,create_tables_2,False)) # Determine fitness of population
            end_test = time.time()
            print('--------------------- Testing End | elapsed time : ',(end_test-start_pop))
            curr_generation_fitness = np.array(curr_generation_fitness).mean(axis=0)# Calculate mean across dimension 2d array
            save_generation_data(curr_generation_fitness,curr_generation,stock_index,create_tables_2)
            curr_selection = selection_operator(curr_generation,curr_generation_fitness) # Select parents from population
            children_list_children_list_text = crossover_operator(curr_selection) # Crossover population
            children_list_children_list_text = mutation_operator(children_list_children_list_text) # mutate population
            curr_generation = assembling_operator(children_list_children_list_text,curr_selection) # generate new population
            create_tables_2 = False
            end_pop = time.time()
            print('------------------------------------------ Whole generation elapsed time : ',(end_pop-start_pop),'| time so far:',(end_pop-start_main))
            print('------------------------------------------------------------------------------------------ New generation | Generation:',id_population)
            # for id_organisms in range(1,num_organisms):
            #     print('test')
        print('------------------------------------------------------------------------------------------ finish | for species :',id_species)
    print('------------------------------------------------------------------------------------------ END |')
    print('-------------------------------------------------------------------------------- Whole Job elapsed time : ',(end_pop-start_main))           
           

    # top = sorted(range(len(curr_generation_fitness_t)), key=lambda i: curr_generation_fitness_t[i])[-2:]
    # print(curr_generation_fitness_t[top[0]],curr_generation_fitness_t[top[1]])

   # save_generation_data([1000,1020],initial_generation,0,create_tables_2)

            ## Creating initial generation
    # initial_generation = creating_initial_generation(2,1,1)

    # curr_generation_fitness = test_generation(initial_generation,0)
    # print(curr_generation_fitness)



        

    # test_organisms_fitness = test_fitness(test_organisms)









## Handles Arg and useage   
def main_manager():
    if(sys.argv[1]=='help' and len(sys.argv) > 1):print(open("db_manager_help.txt", "r").read())
    elif(sys.argv[1]=='analysis' and len(sys.argv) <= 10):average_based_indicators_analysis()
    elif(sys.argv[1]=='genes' and len(sys.argv) <= 10):average_based_indicators_genes_table()
    elif(sys.argv[1]=='algorithm' and len(sys.argv) <= 10):average_based_indicators_genetic_algorithm()
    else:print("Incorrect/Unrecognized Arguments, Type 'db_manager.py help' for help")

if __name__ == "__main__":
    print("BASE_DIR = ",BASE_DIR)
    print("Today_Da = ",pd.to_datetime('now'),'\n')
    main_manager()



    # ## takes data from database and makes it in to backtrader usable object of Indicator type
    # ## Helper fuctions
    # def average_based_indicators_helper_coverter(Indicator_object,
    #     PD_indicator_object,
    #     Parameters_dic):
    #     ## Calculating procedurally-generated name
    #     Indicator_name = PD_indicator_object['Indicator_str'].replace('ind','').lower()+'_'+str("_".join(map(str, Parameters_dic.values()))).replace('.','p')
    #     ## Setting data to it's line in backtrader
    #     setattr(getattr(getattr(Indicator_object,"lines"),Indicator_name),"array",dataframe[Indicator_name].tolist())
    # def average_based_indicators_json_helper_coverter(Indicator_object,
    #     PD_indicator_object,
    #     Link_Indicator_Parameters_dic,
    #     Curr_Indicator_Parameters_dic,
    #     Indicator_Parameters_keys):

    #     ## If value is a float then will move to reasonable integer but multiplying by float_safety_prim (some number with alot of zeros)
    #     float_safety = 1
    #     if isinstance(Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][0], float):
    #         float_safety = float_safety_prim
    #     if isinstance(Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][1], float):
    #         float_safety = float_safety_prim
    #     if isinstance(Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][2], float):
    #         float_safety = float_safety_prim

    #     ## Will set the min(start), max(end), and amount(freqs) for intra-indicator indicator parameters formations
    #     start = Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][0]*float_safety
    #     end = Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][1]*float_safety
    #     freqs = Link_Indicator_Parameters_dic[Indicator_Parameters_keys[0]][2]*float_safety

    #     ## Creating or adding to curr set of indicator parameters, to make indicator parameters formation
    #     for x in range(int(start),int(end),int(freqs)):
    #         if (float_safety == 1) and not (x == 0):
    #             Curr_Indicator_Parameters_dic[Indicator_Parameters_keys[0]] = int(x)
    #         elif x == 0:
    #             Curr_Indicator_Parameters_dic[Indicator_Parameters_keys[0]] = int(x)
    #         else:
    #             Curr_Indicator_Parameters_dic[Indicator_Parameters_keys[0]] = x/float_safety

    #         ## if more parameters founded, recursively adding another layer to the indicator parameters formation
    #         if len(Indicator_Parameters_keys) > 1:
    #             average_based_indicators_json_helper_coverter(Indicator_object,
    #                 PD_indicator_object,
    #                 Link_Indicator_Parameters_dic,
    #                 Curr_Indicator_Parameters_dic,
    #                 Indicator_Parameters_keys[1:])
    #         else:
    #             average_based_indicators_helper_coverter(Indicator_object,PD_indicator_object,Curr_Indicator_Parameters_dic)
    # ## Main Class
    # class average_based_database_converter(bt.Indicator):
    #     lines = tuple(average_based_indicators_lines)
    #     def once(self, start, end):
    #         for PD_indicator_object in tqdm(average_based_indicators, desc='Database Data converter ', position=0):
    #             if len(list(PD_indicator_object['Parameters_dic'].keys())) > 0 :
    #                 average_based_indicators_json_helper_coverter(self,           #Indicator_object (database_converter Indicator_object)             
    #                     PD_indicator_object,                                    #PD_indicator_object
    #                     PD_indicator_object['Parameters_dic'],                  #Link_Indicator_Parameters_dic
    #                     {},                                                     #Curr_Indicator_Parameters_dic
    #                     list(PD_indicator_object['Parameters_dic'].keys()))     #Indicator_Parameters_keys)                                                 
    #             else:
    #                 average_based_indicators_helper_coverter(self,PD_indicator_object['Indicator_str'],{})


    # cerebro = bt.Cerebro()  # create a "Cerebro" engine instance
    # # cerebro = bt.Cerebro(stdstats=False)

    # # Pass it to the backtrader datafeed and add it to the cerebro
    # data = bt.feeds.PandasData(dataname=dataframe)

    # cerebro.broker.set_cash(1000)
    # cerebro.broker.setcommission(commission=0.01)
    # cerebro.adddata(data)  # Add the data feed
    # cerebro.addstrategy(chromosome_StrategyContainer)  # Add the trading strategy



    # cerebro.addsizer(bt.sizers.PercentSizer, percents=20)  # default sizer for strategies



    # print('Starting Portfolio Value : %0.2f' % cerebro.broker.getvalue())
    # cerebro.run() # run it all
    # cerebro.plot() # and plot it with a single command
    # print('Final Portfolio Value : %0.2f' % cerebro.broker.getvalue())
