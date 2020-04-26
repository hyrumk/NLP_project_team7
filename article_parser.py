import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests

from nltk.tokenize import word_tokenize

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

import time


# PIP INSTALLS: selenium, bs4

url = 'http://www.prnewswire.com/news-releases/tata-consultancy-services-reports-broad-based-growth-across-markets-marks-steady-fy17-300440934.html'

def extract_text_from_url(url):
    '''
    Extracts useful texts from provided url
    Step 1.
        Remove texts under irrelevant parent html tags

    <TODO>
    Step 2.
        Regex stripping
    
    Step 3. 
        Contextual Relevance Filter
        --> Comparison of article with other articles of different subject 
        --> temporal similarity + nominal differentiation
        -->

    :param url: REQUIRED
    :return: group of texts extracted from url, delimtted by single space
    '''
    # url = 'http://www.prnewswire.com/news-releases/tata-consultancy-services-reports-broad-based-growth-across-markets-marks-steady-fy17-300440934.html'
    res = requests.get(url)
    html = res.content
    soup = BeautifulSoup(html, 'html.parser')

    text = soup.find_all(text=True)

    output = ''
    blacklist = [
        '[document]',
        'noscript',
        'header',
        'html',
        'meta',
        'head', 
        'input',
        'script',
        'button',
        'tr',
        'table',
        'td',
        'style',
        'footer'
    ]

    # t.parent.name gives parent html tag
    for t in text:
        if t.parent.name not in blacklist:
            output += '{words} '.format(words = t)


    
    token_list = word_tokenize(output)

    return token_list



def date_convert(date_str):
    month_ref = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    result_str =''
    if 'hours ago' in date_str:
        return str(datetime.date(datetime.now()))
    else:
        # Tokenize the date
        #   ex) Apr 24, 2020 --> [Apr, 24, 2020]
        #   Rearrange and format to 'YYYY - MM - DD'
        date_str.replace(',', '')
        tmp_list = date_str.split()

        result_str = result_str + tmp_list[2]

        month_index = month_ref.index(tmp_list[0])
        if len(str(month_index + 1)) == 1:
            result_str = result_str + '-0' + str(month_index + 1) + '-'
        else:
            result_str = result_str + '-' + str(month_index + 1) + '-'
        
        result_str = result_str + tmp_list[1]

        result_str.replace(',', '')
        return result_str




def urls_from_domain(company_name, default_url='https://www.investing.com'):
    '''
    Extracting articles from provided domain, using the filter results based on the company_name

    :param company_ticker: ex) AMZN for Amazon
    :param start_date: 'YYYY - MM - DD' format
    :param end_date: 'YYYY - MM - DD' format


    :return: [date, url] pair ?
    '''
    
    #TODO: have to verify if company_ticker is valid.
    domain_frame = 'https://www.investing.com/search/?q={keyword}&tab=news'.format(keyword = company_name)

    url_triplet = []

    #scroll to bottom of url
    driver = webdriver.Chrome()
    driver.get(domain_frame)

    lenOfPage = driver.execute_script(
        "window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
    match = False
    while(match == False):
        lastCount = lenOfPage
        time.sleep(10)
        lenOfPage = driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
        

        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        link_data = []
        date_data = []
        for div in soup.find_all('div', {"class": "textDiv"}):
            #TODO: get time of the articles

            href_child = div.find('a', href=True)
            link = default_url + href_child['href']

            # check for duplicates
            if link not in link_data:
                link_data.append(link)

        for time_tag in soup.find_all('time', {'class': 'date'}):
            date = time_tag.text

            # check for duplicates
            #if date not in date_data:
            date_data.append(date)
        

        if lastCount == lenOfPage:
            match = True

    for i in range(len(link_data)):
        tmp_ele = []
        tmp_ele.append(link_data[i])
        date_str = date_convert(date_data[i])
        date_str.replace(',', '')
        tmp_ele.append(date_str)

        #check if company_name in title/url 
        if company_name in link_data[i]:
            tmp_ele.append('True')
        else:
            tmp_ele.append('False')

        url_triplet.append(tmp_ele)

    return url_triplet




#print(extract_text_from_url(url))

url_list = urls_from_domain('apple')

with open('listfile.txt', 'w') as filehandle:
    filehandle.truncate(0)
    for listitem in url_list:
        filehandle.write('%s , %s , %s\n' % (listitem[0], listitem[1], listitem[2]))


#print(extract_text_from_url(url_list[0][0]))
