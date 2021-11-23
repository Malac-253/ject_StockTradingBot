# average_based_indicators_analysis.py

#By Malachi I Beerram
#GitHub @malac-253
#Started:		07/19/21
#Last Updated:	07/19/21

## Arguments Check
import sys
# print(">average_based_indicators_analysis.py")
# print(f"Arguments count: {len(sys.argv)}")
# for i, arg in enumerate(sys.argv):
#     print(f"Argument {i:>6}: {arg}")

# print("Imports loading ...")

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
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
from itertools import permutations
import gc
import time
import multiprocessing as mp

# print("Imports loaded  :)")

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

        #print('\t\tinit c : Indicator_analysis_helper_json_init',Curr_Indicator_Parameters_dic,Indicator_Parameters_keys,int(start),int(end),int(freqs))

        #print("\t",int(start),int(end),int(freqs))

        for x in range(int(start),int(end),int(freqs)):
            if (float_safety == 1) and not (x == 0):
                Curr_Indicator_Parameters_dic[Indicator_Parameters_keys[0]] = int(x)
            elif x == 0:
                Curr_Indicator_Parameters_dic[Indicator_Parameters_keys[0]] = int(x)
            else:
                Curr_Indicator_Parameters_dic[Indicator_Parameters_keys[0]] = x/float_safety

            #print('init 1: ',Indicator_object['Indicator_str'].lower()+'_'+str("_".join(map(str, Curr_Indicator_Parameters_dic.values()))))
            #print('init 1:',Indicator_Parameters_keys,(len(Indicator_Parameters_keys) > 1))

            if len(Indicator_Parameters_keys) > 1:
                #print('\tinit - : ',Indicator_Parameters_keys[1:],Curr_Indicator_Parameters_dic,len(Indicator_Parameters_keys[1:]) > 1)

                Indicator_analysis_helper_json_init(Indicator_analysis_obj,Indicator_object,
                    Link_Indicator_Parameters_dic,
                    Curr_Indicator_Parameters_dic,
                    Indicator_Parameters_keys[1:])

                #print('\tinit -2 :' ,Indicator_Parameters_keys[1:], Curr_Indicator_Parameters_dic,len(Indicator_Parameters_keys[1:]) > 1)
            else:
                #print('init 2: ',Indicator_object['Indicator_str'].lower()+'_'+str("_".join(map(str, Curr_Indicator_Parameters_dic.values()))))
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

        



def average_based_indicators_genes_table():
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

    ## average_based_indicators_lines Helper
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

    ## The Indicator analysis array
    average_based_indicators = json.load(open('average_based_indicators.json'))# Opening JSON file
    average_based_indicators_lines = average_based_indicators_lines_coverter(average_based_indicators)
   
    ## Using all stock
    
    # for symbol in tqdm(symbols, desc='Creating CrossOver Genes', position=2):
    #     query = """
    #             SELECT * FROM average_based_indicators_analysis_table
    #                     WHERE symbol = '"""+symbol+"""' 
    #                     AND date > '""" +start+"""' 
    #                     AND date < '""" +end+"""' 
    #                 ORDER BY symbol ASC, date ASC
    #             """

    #     data = engine.execute(query)
    #     dataframe = pd.DataFrame(data.fetchall())
    #     dataframe.columns = data.keys()
    #     dates = dataframe['date'].tolist()
    #     dataframe.set_index('date', inplace=True)

        

    #     float_safety_prim = 10000
    #     plot_spacing = 1#5

    crossover_groups =list(permutations(average_based_indicators_lines, 2))
    #https://stackoverflow.com/questions/13784192/creating-an-empty-pandas-dataframe-then-filling-it
    create_tables = False
    if(sys.argv[3] == "create"):
        create_tables = True
    # counter2 = 0
    # total = len(crossover_groups)
    pool = mp.cpu_count() - 1
    #if pool == 0: 
    print(" >> mp.cpu_count() = ",pool)
    
    #https://stackoverflow.com/questions/41920124/multiprocessing-use-tqdm-to-display-a-progress-bar
    #https://tutorialedge.net/python/python-multiprocessing-tutorial/
    #https://www.codementor.io/@satwikkansal/python-practices-for-efficient-code-performance-memory-and-usability-aze6oiq65
    #https://docs.python.org/2/library/multiprocessing.html
    #https://medium.datadriveninvestor.com/how-does-memory-allocation-work-in-python-and-other-languages-d2d8a9398543
    if(create_tables):
        for lines in average_based_indicators_lines:
            engine.execute('DROP TABLE IF EXISTS average_based_indicators_genes_table_%s;'% (str(lines[1])))## Firing query statment
            query = """
                    CREATE TABLE average_based_indicators_genes_table_%s AS
                        SELECT date, symbol
                        FROM average_based_indicators_analysis_table;
                    """% (str(lines[1]))
            engine.execute(query) ## Firing query statment
            engine.execute('ALTER TABLE average_based_indicators_genes_table_%s ADD PRIMARY KEY (symbol, date);'% (str(lines[1])))## Firing query statment
    
    create_tables = False
    base = len(average_based_indicators_lines)
    mass = len(crossover_groups)
    print(mass/base)
    for i in tqdm(range(int(mass/base)),desc='Creating CrossOver Genes Boss', position=1):
        i_sub = i * base
        with mp.Pool(pool) as p:
            r = list(tqdm(p.imap(for_process,crossover_groups[i_sub:i_sub+base]),total=len(crossover_groups[i_sub:i_sub+base]),desc='Creating CrossOver Manager %s/%s'%(str(i+1),(mass/base)), position=0))
        time.sleep(2)

    # for crossover_group in tqdm(crossover_groups[0:2], desc=('Creating CrossOver Genes for %s'%str(symbols)), position=0):
    #     result = pool.map(my_func, create_tables, )



        #         counter = 0
        #         ##adding in new COLUMN here
        #         for gene in tqdm(gene_df_sub, desc='Recording Genes data for ['+str(counter2)+'/'+str(total)+']'+gene_name+' for '+symbol, position=0):
        #             query = ("""
        #                     UPDATE average_based_indicators_genes_table
        #                         SET """+gene_name+""" = '"""+str(gene)+"""' 
        #                         WHERE date = '"""+str(dates[counter])+"""'
        #                             AND symbol ='"""+symbol+"""';
        #                     """)
        #             engine.execute(query)
        #             counter = counter + 1 
        #         del counter
        #         gc.collect()

        #     del crossover_group
        #     del query 
        #     del gene_name
        #     del gene_df_sub
        #     gc.collect()
        #     counter2 = counter2 + 1 
        # del crossover_groups
        # del data
        # del dataframe
        # del dates
        # del symbol
        # gc.collect()

   

def for_process(crossover_group):
    gene_name = ("crossover_%s"%("_X_".join([crossover_group[0][1],crossover_group[1][1]])))
    for symbol in symbols:
    ## Great sql work Here
        
        ## https://medium.datadriveninvestor.com/how-does-memory-allocation-work-in-python-and-other-languages-d2d8a9398543
        ## Make new column for data
        if(sys.argv[3] == "create"):
            query = ('ALTER TABLE IF EXISTS ONLY %s ADD COLUMN %s %s' % (("average_based_indicators_genes_table_%s" % crossover_group[0][1]), gene_name , "DOUBLE PRECISION"))
            engine.execute(query) ## Firing query statment

        ## Add data to column 
        #https://stackoverflow.com/questions/42729849/unexpected-deadlocks-in-postgresql-while-using-psycopg2
        #https://www.postgresql.org/docs/9.5/runtime-config-compatible.html
        # set lock_timeout = 100;
        # SET LOCAL synchronize_seqscans = off;
        # https://dba.stackexchange.com/questions/200478/update-if-no-lock-in-postgres
        # https://www.pgcasts.com/episodes/the-skip-locked-feature-in-postgres-9-5
        # http://www.sqlines.com/oracle-to-mysql/skip_locked
        # https://www.2ndquadrant.com/en/blog/what-is-select-skip-locked-for-in-postgresql-9-5/
        # SELECT %s
        #         FROM average_based_indicators_genes_table_%s
        #             WHERE average_based_indicators_genes_table_%s.symbol = '%s'
        #             AND average_based_indicators_genes_table_%s.date = average_based_indicators_genes_table_%s.date
        #             FOR UPDATE NOWAIT;
        # https://stackoverflow.com/questions/20033816/protect-against-parallel-transaction-updating-row
        #https://www.postgresql.org/docs/9.1/sql-select.html - "E will wait for the o"
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

## Handles Arg and useage   
def main_manager():
    if(sys.argv[1]=='help' and len(sys.argv) > 1):print(open("db_manager_help.txt", "r").read())
    elif(sys.argv[1]=='analysis' and len(sys.argv) <= 10):average_based_indicators_analysis()
    elif(sys.argv[1]=='genes' and len(sys.argv) <= 10):average_based_indicators_genes_table()
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
