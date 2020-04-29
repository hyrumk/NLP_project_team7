import FinanceDataReader as fdr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def stock_price_data(company_ticker, start_date = 'start', end_date = 'end', price_type = 'Open'):
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
    Filters the stock price data of a given company separated in a given interval.
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
    data_index = [price_data.index[0]]
    data_element = [price_data[0]]
    next_date = price_data.index[0] + timedelta(days = interval)
    for date in price_data.index:
        if date < next_date:
            continue
        elif date == next_date:
            data_index.append(date)
            data_element.append(price_data[date])
            next_date += timedelta(days = interval)
        else:
            data_index.append(date)
            data_element.append(price_data[date])
            while next_date < date:
                next_date += timedelta(days = 1)
            next_date = date + timedelta(days = interval)

    filtered_data = pd.Series(data_element, index = data_index)

    return filtered_data



def stock_price_label(company_ticker, interval = 1, percentage_rate = 3,
                      start_date = 'start', end_date = 'end', price_type = 'Open'):
    '''
    Returns the label to feed as an input for a classifier.

    The label will have three different types,
    [1,0,0] when the stock price rised by over 3% than the day before.(percentage depends on given percentage_rate)
    [0,1,0] when the stock price change is within the range of -3% ~ +3% (depends on given percentage_rate)
    [0,0,1] when the stock price fell more than 3% than the day before.(depends on given percentage_rate)
    **The returned data will not include the data for the given start_date,
        as there is no price data from previous day to compare.**

    :param company_ticker: refer to stock_price_interval()
    :param interval: refer to stock_price_interval()
    :param start_date: refer to stock_price_interval()
    :param end_date: refer to stock_price_interval()
    :param price_type:
    :return: labeled data in pandas series (index: Timestamp, element: list)
    '''
    percentage_rate *= 0.01
    stock_price_data = stock_price_interval(company_ticker, interval, start_date, end_date, price_type)
    data_index = []
    data_label = []
    previous_price = stock_price_data[0]
    for date in stock_price_data.index[1:]:
        data_index.append(date)
        rate = stock_price_data[date]/previous_price
        previous_price = stock_price_data[date]
        if rate > 1 + percentage_rate:
            data_label.append([1,0,0])
        elif rate < 1 - percentage_rate:
            data_label.append([0,0,1])
        else:
            data_label.append([0,1,0])

    labeled_data = pd.Series(data_label, index = data_index)
    return labeled_data
