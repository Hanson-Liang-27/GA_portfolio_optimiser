import pandas as pd 
import numpy as np
import pandas_datareader.data as web

def fetch_data(
    tickers: list,
    start_date: str = '2019-02-01',
    end_date: str = '2020-12-01') -> pd.DataFrame:
    """
    Load stock data from Yahoo.

    Parameters
    ----------
    tickers : list of stock market tickers
    start_date : date where data extraction starts  
    end_date : date where data extraction ends  

    Returns
    -------
    portfolio : Dataframe with daily stock data
    """

    portfolio = pd.DataFrame()
    
    for t in tickers:
        portfolio[t] = web.DataReader(t, data_source = 'yahoo', start = start_date, end = end_date)['Adj Close']
        
    portfolio.columns = tickers
    
    return portfolio 
