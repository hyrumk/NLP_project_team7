from stock_data import stock_price_label
from datetime import timedelta
from article_parser import extract_text_from_url, urls_search_by_keyword
import string
import pandas as pd
import datetime as dt
import os.path

#install lxml v 4.5.0
'''
In 'Data Storage' folder
    - listfile_{keyword}.txt: A list of urls for a given keyword
        saved through urls_search_by_keyword or text_from_url_list
    - keyword_{keyword}.pkl: A pkl file that consists the pandas.Series of a text data for a given keyword.
        saved through store_data when a data from 'text_from_url_list' and data_type = 'keyword' given as an input.
    - price_{company_ticker}.pkl: A pkl file that has the price data of a given company_ticker.
        saved through store_data when a data from stock_price_interval or stock_price_data is given.
    - label_{company_ticker}.pkl: A pkl file that has the label data of a given company_ticker.
        saved through store_data when a data from stock_price_label is given.
'''



COMPANY_TICKER_INPUT = 'AAPL'
KEYWORD_INPUT        = 'apple'
INTERVAL             = 7
START_DATE           = '2010-01-01'


# 혹시라도 f.readlines()에서 문제 생기면 f = open의 3rd parameter -> encoding = 'UTF-8' 추가
def text_from_url_list(keyword, in_title = False):
    '''
    Receives the url to make a text data in pandas.Series format.
    If a listfile of a given keyword doesn't exist, it will create a new one.

    :param keyword: keyword string
    :param in_title: if True, only take news with keyword in the title
    :return: a text data with datatime index (pandas.Series)
            (index: datatime, element: list of texts)
            'NEXTTEXT' in the text data indicates that it's the boundary between different articles
            when there are multiple articles on certain date.
    '''
    series_index = []
    series_text = []
    modified_keyword = keyword.translate({ord(c): None for c in string.whitespace}).lower()
    txt_file = './Data Storage/listfile_{}.txt'.format(modified_keyword)
    file_exists = os.path.exists(txt_file)
    if not file_exists:
        urls_search_by_keyword(keyword)
    f = open(txt_file, 'rt')
    text = f.readlines()
    for line in text:
        text_data = ''
        url_triplet = [element.strip() for element in list(line.split(', ')) if element != '']
        try:
            text_data = extract_text_from_url(url_triplet[0])
        except:
            continue
        text_data = ' '.join(text_data)
        time_list = url_triplet[1].split('-')
        time = dt.datetime(int(time_list[0]), int(time_list[1]), int(time_list[2]))
        if in_title and url_triplet[2] == 'True':
            if time in series_index:
                ind = series_index.index(time)
                series_text[ind].append(text_data) # + ' NEXTTEXT ' #
            else:
                series_index.append(time)
                series_text.append([text_data])
        elif in_title and url_triplet[2] != 'True':
            continue
        else:
            if time in series_index:
                ind = series_index.index(time)
                series_text[ind].append(text_data) #+ ' NEXTTEXT ' #
            else:
                series_index.append(time)
                series_text.append([text_data])
    f.close()
    text_series = pd.Series(series_text, index = series_index).sort_index()
    return text_series


def title_from_url_list(keyword):
    '''
    receives titles from the news with a title that include the given keyword.

    :param keyword: keyword string
    :return: a title data with datatime index (pandas.Series)
            (index: datatime, element: list of texts)
    '''
    series_index = []
    series_title = []
    modified_keyword = keyword.translate({ord(c): None for c in string.whitespace}).lower()
    txt_file = './Data Storage/listfile_{}.txt'.format(modified_keyword)
    file_exists = os.path.exists(txt_file)
    if not file_exists:
        urls_search_by_keyword(keyword)
    f = open(txt_file, 'rt')
    text = f.readlines()
    for line in text:
        text_data = ''
        url_triplet = [element.strip() for element in list(line.split(', ')) if element != '']
        if url_triplet[2] == 'True':
            url = url_triplet[0]
            title = url.split('/')[-1].split('-')
            if str.isdecimal(title[-1]):
                del title[-1]
            text_data = title
            time_list = url_triplet[1].split('-')
            time = dt.datetime(int(time_list[0]), int(time_list[1]), int(time_list[2]))
            if time in series_index:
                ind = series_index.index(time)
                series_title[ind] += text_data # + 'NEXTTEXT'
            else:
                series_index.append(time)
                series_title.append(text_data)
        else:
            continue
    f.close()

    title_series = pd.Series(series_title, index = series_index).sort_index()
    return title_series


def relevant_news_from_url_list(keyword, relevance_keyword = 'technology'): # technology section
    series_index = []
    series_text = []
    modified_keyword = keyword.translate({ord(c): None for c in string.whitespace}).lower()
    txt_file = './Data Storage/listfile_{}.txt'.format(modified_keyword)
    file_exists = os.path.exists(txt_file)
    if not file_exists:
        urls_search_by_keyword(keyword)
    f = open(txt_file, 'rt')
    text = f.readlines()
    for line in text:
        text_data = ''
        url_triplet = [element.strip() for element in list(line.split(', ')) if element != '']
        if relevance_keyword in url_triplet[0]:
            try:
                text_data = extract_text_from_url(url_triplet[0])
            except:
                continue
        else:
            continue
        time_list = url_triplet[1].split('-')
        time = dt.datetime(int(time_list[0]), int(time_list[1]), int(time_list[2]))
        if time in series_index:
            ind = series_index.index(time)
            series_text[ind].append(text_data) #+ ' NEXTTEXT ' #
        else:
            series_index.append(time)
            series_text.append([text_data])
    f.close()
    text_series = pd.Series(series_text, index = series_index).sort_index()
    return text_series

def check_date_range(series_data):
    '''
    returns the starting and the ending dates of the given data.
    '''
    start = str(series_data.index[0])
    end = str(series_data.index[-1])
    return start, end



def newsnumber_by_date(keyword):
    series_index = []
    series_numbernews = []
    modified_keyword = keyword.translate({ord(c): None for c in string.whitespace}).lower()
    txt_file = './Data Storage/listfile_{}.txt'.format(modified_keyword)
    file_exists = os.path.exists(txt_file)
    if not file_exists:
        urls_search_by_keyword(keyword)
    f = open(txt_file, 'rt')
    text = f.readlines()
    for line in text:
        url_triplet = [element.strip() for element in list(line.split(', ')) if element != '']
        time_list = url_triplet[1].split('-')
        time = dt.datetime(int(time_list[0]), int(time_list[1]), int(time_list[2]))
        if time in series_index:
            ind = series_index.index(time)
            series_numbernews[ind] += 1  # + 'NEXTTEXT'
        else:
            series_index.append(time)
            series_numbernews.append(1)
    f.close()
    numbernews_series = pd.Series(series_numbernews, index = series_index).sort_index()
    return numbernews_series


def merge_price_newsnumber(newsnumber_data, label_data):
    date_index = [label_data.index[0]]

    first_number = 0
    text_index = 0
    for date in newsnumber_data.index:
        if date <= label_data.index[0]:
            first_number += newsnumber_data[date]
            text_index += 1
        else:
            break
    data = [(first_number, label_data[0])]

    for date in label_data.index[1:]:
        text_to_add = 0
        label = label_data[date]
        while text_index != len(newsnumber_data) and newsnumber_data.index[text_index] <= date:
            text_to_add += newsnumber_data[text_index]
            text_index += 1

        if (text_to_add != 0):
            date_index.append(date)
            data.append((text_to_add, label_data[date]))
        else:
            continue

    return pd.Series(data, index=date_index)




def merge_price_text(text_data, label_data):
    '''
    Merges text and label data to create.
    Won't append any data if text data for the given date range doesn't exist.

    :param text_data: text data in pandas.Series
    :param label_data: label data in pandas.Series
    :return: pandas.Series consisted of tuples of text list and label by date
    '''
    date_index = [label_data.index[0]]
    first_text = []
    text_index = 0
    for date in text_data.index:
        if date <= label_data.index[0]:
            first_text += text_data[date]
            text_index += 1
        else:
            break
    data = [(first_text, label_data[0])]

    for date in label_data.index[1:]:
        text_to_add = []
        label = label_data[date]
        while text_index != len(text_data) and text_data.index[text_index] <= date:
            text_to_add += text_data[text_index]
            text_index += 1

        if (text_to_add != []):
            date_index.append(date)
            data.append((text_to_add, label_data[date]))
        else: continue

    return pd.Series(data, index = date_index)
####MODIFY above return when needed!####


def store_data(series_data, title, data_type = 'price'):
    '''
    stores the given pandas.Series data into a pickle file

    :param series_data: the data to be stored in a picke file
    :param title: keyword if data_type is 'keyword', else company ticker
    :param data_type: 'price' stock_price_interval or stock_price_data from stock_data
                      'label' stock_price_label from stock_data
                    'keyword' from text_from_url_list from data_collector
    return: None
    '''
    dtype = ['price', 'label', 'keyword']
    if data_type in dtype:
        if data_type in dtype[:2]:
            title = title.upper()
        series_data.to_pickle("./Data Storage/{}_{}.pkl".format(data_type, title))




def load_data(title, data_type = 'price'):
    '''
    loads the data for a given title and data_type if it exists.

    :param title: keyword if data_type is 'keyword', else company ticker
    :param data_type: 'price' stock_price_interval or stock_price_data from stock_data
                      'label' stock_price_label from stock_data
                    'keyword' from text_from_url_list from data_collector
    return: loaded pandas.Series for a given title and data_type
    '''
    file_name = "./Data Storage/{}_{}.pkl".format(data_type, title)
    file_exists = os.path.exists(file_name)
    dtype = ['price', 'label', 'keyword']
    loaded_data = None
    if data_type in dtype and file_exists:
        if data_type in dtype[:2]:
            title = title.upper()
        loaded_data = pd.read_pickle(file_name)
    elif not data_type in dtype:
        raise NameError("Wrong data_type. Choose among 'price', 'label', or 'keyword'.")
    else:
        raise NameError("No such file exists in your directory.")

    return loaded_data


'''
txt_series = text_from_url_list(KEYWORD_INPUT)
label_series = stock_price_label(COMPANY_TICKER_INPUT, interval = INTERVAL, start_date= START_DATE)
txt_series.to_pickle("./Data Storage/keyword_{}.pkl".format(KEYWORD_INPUT.lower()))
label_series.to_pickle("./Data Storage/label_{}.pkl".format(COMPANY_TICKER_INPUT))
'''

#txt_series = text_from_url_list(KEYWORD_INPUT)
#store_data(txt_series, 'apple', 'keyword')
#txt_series.to_pickle("./Data Storage/keyword_{}.pkl".format(KEYWORD_INPUT.lower()))
#txt= text_from_url_list('Nasdaq')
#store_data('keyword', 'Nasdaq')

#txt_series = text_from_url_list(KEYWORD_INPUT)
#store_data(txt_series, KEYWORD_INPUT, 'keyword')