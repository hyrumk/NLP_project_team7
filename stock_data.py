import FinanceDataReader as fdr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def stock_price_data(company_ticker, start_date, end_date, price_type):
    '''
    Gives the stock data of a given company in chronological order.

    param info in function 'stock_price_interval'
    :return: price data of a given price_type in chronological order (pandas.core.series.Series)
            price data type = numpy.float64
            date data type = pandas.Timestamp
    '''
    if start_date == 'start' and end_date == 'end':
        price_data = fdr.DataReader(company_ticker)[price_type]
    elif end_date == 'end':
        price_data = fdr.DataReader(company_ticker, start_date)[price_type]
    elif start_date == 'start':
        price_data = None
    else:
        price_data = fdr.DataReader(company_ticker, start_date, end_date)[price_type]

    return price_data


def stock_price_interval(company_ticker, interval, start_date = 'start', end_date = 'end', price_type = 'Open'):
    '''
    Filters the stock price data of a given company in a given interval.
    e.g. stock_price_interval('AAPL', 7, '2010-10-10', '2020-03-20') will return price data of 2010-10-10 ~ 2020-03-20 in 7 day interval
        stock_price_interval('AAPL', 2) will return the entire price data in 2 day interval.

    Not giving start_date and end_date input gives the entire price data available.
    Giving only the end_date will return None. (the library doesn't support such function)
    Giving only the start_date will return the price data of (start_date ~ most recent market day)
    Giving both the start_date and the end_date will return the price data of (start_date ~ end_date)

    :param company_ticker: a ticker of a company e.g. APPLE --> 'AAPL' (ALWAYS REQUIRED)
    :param interval: number of days
    :param start_date: 'YYYY - MM - DD' format
    :param end_date: 'YYYY - MM - DD' format
    :param price_type: 'Open', 'Close', 'High', 'Low'
    :return:
    '''

    price_data = stock_price_data(company_ticker, start_date, end_date, price_type)
    filtered_data = [(price_data.index[0], price_data[0])]
    next_date = price_data.index[0] + timedelta(days = interval)
    for index, date in enumerate(price_data.index):
        if date < next_date:
            continue
        elif date == next_date:
            filtered_data.append((date, price_data[index]))
            next_date += timedelta(days = interval)
        else:
            filtered_data.append((date, price_data[index]))
            delay = date - next_date
            if timedelta(days = interval) - delay < timedelta(days = 1):
                next_date = date + timedelta(days = 1)
            else:
                next_date = date + (timedelta(days = interval) - delay)

    return filtered_data


