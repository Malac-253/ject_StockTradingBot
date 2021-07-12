import pandas_datareader as web
from datetime import datetime
import pandas as pd


start = datetime(2020,1,1)
end = datetime(2021,6,24) #YYYY,MM,DD
#stock = ['GOOG','TSLA','MMM']
stock = ['GOOG']

df = web.DataReader(stock,'yahoo',start,end)

#df = pd.DataFrame(, columns=['a', 'b', 'c', 'd', 'e'])
#df.reset_index(inplace=True,drop=False)

#df.to_excel(f'src/historical_data/stockdata_{stock[0]}.xlsx')
#df.to_excel(f'src/historical_data/stockdata_{stock[0]}.xlsx')
df.to_csv(f'src/historical_data/stockdata_{stock[0]}.csv')

#daily_returns = df['Adj Close'].pct_change()
#monthly_returns = df['Adj Close'].resample('M').ffill().pct_change()

#daily_returns.to_excel(f'src/historical_data/daily_returns.xlsx')
#monthly_returns.to_excel(f'src/historical_data/monthly_returns.xlsx')

#daily_returns.to_csv(f'src/historical_data/daily_returns_stockdata_{stock[0]}.csv')
#monthly_returns.to_csv(f'src/historical_data/monthly_returns_stockdata_{stock[0]}.csv.')
#print(df.head())
#print("testing")
print("testing")

first_column = df.columns[0]
print("First Column Of Dataframe: ") 
print(first_column,"\n")


row_1=df.iloc[0]

print("The First Row of the DataFrame is:")
print(row_1,"\n")

print("The DataFrame is:")
print(df,"\n")







import plotly.graph_objects as go


#df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

# fig = go.Figure(data=[go.Candlestick(x=df['Attributes'], #df['Date'],
#                 open=df['Open'],
#                 high=df['High'],
#                 low=df['Low'],
#                 close=df['Close'])])

# fig.show()
