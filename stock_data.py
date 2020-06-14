import FinanceDataReader as fdr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def stock_price_data(company_ticker, start_date = 'start', end_date = 'end', price_type = 'Open'):
    '''
    Gives the stock data of a given company in chronological order.
    for market index: (in company_ticker param)
        IXIC = NASDAQ, DJI = Dow Jones, US500 = S&P 500,
        KS11 = KOSPI, KQ11 = KOSDAQ
        JP225 = Nikkei 225, HSI = Hang Seng, SSEC = Shanghai



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


def market_price_data(market = 'Nasdaq', start_date = 'start', end_date = 'end', price_type = 'Open'):
    '''


    :return: index data of a given market
    '''

    if start_date == 'start' and end_date == 'end':
        price_data = fdr.DataReader(market)[price_type]
    elif end_date == 'end':
        price_data = fdr.DataReader(market, start_date)[price_type]
    elif start_date == 'start':
        price_data = None
    else:
        price_data = fdr.DataReader(market, start_date, end_date)[price_type]

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


# For rate in relation to market average
def market_stock_growth_interval(company_ticker, market, interval = 1,
                                 start_date = 'start', end_date = 'end',
                                                    price_type = 'Open'):
    '''
    A list of difference between price rate of the market and the company.
    e.g. Apple price: 1 --> 1.5, NASDAQ index: 10 --> 12
        Apple rate: 50%, NASDAQ rate: 20%
        result = 50 - 20 = 30%


    '''
    market_data = stock_price_interval(market, interval, start_date, end_date, price_type)
    stock_data = stock_price_interval(company_ticker, interval, start_date, end_date, price_type)

    date = [stock_data.index[i] for i, price in enumerate(stock_data[:-1])]
    rate = [(stock_data[i+1]/price - 1) * 100 for i, price in enumerate(stock_data[:-1])]
    #rate = [(stock_data[i + 1]/price - market_data[i + 1]/market_data[i])*100 for i, price in enumerate(stock_data[:-1])]

    rate_data = pd.Series(rate, index = date)

    return rate_data

'''
Label Type


stock_price_label : 상승/하락률이 특정 threshold를 넘어가는지에 따라 label 지정
stock_price_label2 : x일 연속으로 상승/하락하는지에 따라 label 지정 (ex. 3일 연속 상승 --> [1,0,0])
stock_price_label3 : 시장 지수 (market index) 와 비교해 상승/하락률이 특정 threshold를 넘어가는지에 따라 label 지정
stock_price_label_binary : binary label ([1,0,0] & [0,0,1]) 단순 주가의 상승/하락에 따라 label 지정
stock_price_label_binary2 : 시장 지수 (market index) 와 비교해 상승/하락에 따라 label 지정


'''


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
    :param percentage_rate: the percentage threshold to consider as increase/decrease
    :param start_date: refer to stock_price_interval()
    :param end_date: refer to stock_price_interval()
    :param price_type:
    :return: labeled data in pandas series (index: Timestamp, element: list)
    '''
    percentage_rate *= 0.01
    stock_price_data = stock_price_interval(company_ticker, interval, start_date, end_date, price_type)
    data_index = []
    data_label = []
    for i, price in enumerate(stock_price_data[:-1]):
        date = stock_price_data.index[i]
        next_date = stock_price_data.index[i + 1]
        data_index.append(date)
        rate = stock_price_data[next_date]/price
        if rate > 1 + percentage_rate:
            data_label.append([1,0,0])
        elif rate < 1 - percentage_rate:
            data_label.append([0,0,1])
        else:
            data_label.append([0,1,0])

    labeled_data = pd.Series(data_label, index = data_index)
    return labeled_data


def stock_price_label2(company_ticker, interval = 1, consecutive = 3,
                      start_date = 'start', end_date = 'end', price_type = 'Open'):
    '''
    2nd way of labeling the price. Labels based on the consecutive trend of the price.

    e.g. when interval = 2 & consecutive = 3,
    [1,0,0] when stock 'Open' price increased for 3 * '2' days in a row.
    [0,0,1] when stock 'Open' price decreased for 3 * '2' days in a row.
    [0,1,0] else.

    :param company_ticker: refer to stock_price_interval()
    :param interval: refer to stock_price_interval()
    :param consecutive: the consecutive ticks to be considered an increase/decrease
    :param start_date: refer to stock_price_interval()
    :param end_date: refer to stock_price_interval()
    :param price_type:
    :return: labeled data in pandas series (index: Timestamp, element: list)
    '''
    stock_price_data = stock_price_interval(company_ticker, interval, start_date, end_date, price_type)
    data_index = []
    data_label = []
    for i, price in enumerate(stock_price_data[:-consecutive]):
        date = stock_price_data.index[i]
        next_date = stock_price_data.index[i + 1]
        data_index.append(date)
        increase, decrease = 0, 0
        if stock_price_data[i] < stock_price_data[i + 1]: # increase = 0 if next day price lower than the previous one even once
            increase = 1
            for j in range(consecutive):
                if stock_price_data[i + j] > stock_price_data[i + j + 1]:
                    increase = 0
                    break
        else: # decrease = 0 if next day price higher than the previous one even once
            decrease = 1
            for j in range(consecutive):
                if stock_price_data[i + j] < stock_price_data[i + j + 1]:
                    decrease = 0
                    break
        if increase == 1:
            data_label.append([1,0,0])
        elif decrease == 1:
            data_label.append([0,0,1])
        else:
            data_label.append([0,1,0])

    labeled_data = pd.Series(data_label, index = data_index)
    return labeled_data


def stock_price_label3(company_ticker, market, interval = 1, percentage_rate = 2, start_date = 'start',
                                            end_date = 'end', price_type = 'Open'):
    rate_list = market_stock_growth_interval(company_ticker, market, interval,
                                             start_date, end_date, price_type)
    data_index = [date for date in rate_list.index]
    data_label = []
    for rate in rate_list:
        if rate >= percentage_rate:
            data_label.append([1,0,0])
        elif rate <= -percentage_rate:
            data_label.append([0,0,1])
        else:
            data_label.append([0,1,0])

    labeled_data = pd.Series(data_label, index = data_index)
    return labeled_data













def stock_price_label_binary(company_ticker, interval = 1,
                      start_date = 'start', end_date = 'end', price_type = 'Open'):
    '''
    Returns the label to feed as an input for a classifier.

    The label will have three different types,
    [1,0,0] when the stock price rised by over 3% than the day before.(percentage depends on given percentage_rate)
    [0,1,0] when the stock price change is within the range of -3% ~ +3% (depends on given percentage_rate)
    [0,0,1] when the stock price fell more than 3% than the day before.(depends on given percentage_rate)
    e.g. if the stock price on 2010-01-01 was 1 and 2 on 2010-01-02, the label of '2010-01-01' will be [1,0,0].
    **The returned data will not include the data for the given start_date,
        as there is no price data from previous day to compare.**

    :param company_ticker: refer to stock_price_interval()
    :param interval: refer to stock_price_interval()
    :param start_date: refer to stock_price_interval()
    :param end_date: refer to stock_price_interval()
    :param price_type:
    :return: labeled data in pandas series (index: Timestamp, element: list)
    '''

    stock_price_data = stock_price_interval(company_ticker, interval, start_date, end_date, price_type)
    data_index = []
    data_label = []
    for i, price in enumerate(stock_price_data[:-1]):
        date = stock_price_data.index[i]
        next_date = stock_price_data.index[i + 1]
        data_index.append(date)
        rate = stock_price_data[next_date]/price
        if rate >= 1:
            data_label.append([1,0,0])
        else:
            data_label.append([0,0,1])

    labeled_data = pd.Series(data_label, index = data_index)
    return labeled_data


def stock_price_label_binary2(company_ticker, market, interval = 1, start_date = 'start',
                                            end_date = 'end', price_type = 'Open'):
    rate_list = market_stock_growth_interval(company_ticker, market, interval,
                                             start_date, end_date, price_type)
    data_index = [date for date in rate_list.index]
    data_label = [[1,0,0] if rate >= 0 else [0,0,1] for rate in rate_list]

    labeled_data = pd.Series(data_label, index = data_index)
    return labeled_data


#print(market_stock_growth_interval('AAPL', 'IXIC', 7, '2010-01-01'))
#print(market_stock_label_binary('AAPL', 'IXIC', 7, '2010-01-01'))