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


def urls_from_domain(company_ticker, default_url='https://finance.yahoo.com/'):
    '''
    Extracting articles from provided domain, using the filter results based on the company_name

    :param domain: Main url of the website
    :param company_ticker: ex) AMZN for Amazon
    :param start_date: 'YYYY - MM - DD' format
    :param end_date: 'YYYY - MM - DD' format


    :return: [date, url] pair ?
    '''
    
    #TODO: have to verify if company_ticker is valid.
    domain_frame = 'https://finance.yahoo.com/quote/{tag}/?p={tag}'.format(tag = company_ticker)

    #form a url list
    url_list = []

    #scroll to bottom of url
    driver = webdriver.Chrome()
    driver.get(domain_frame)

    # for some reason have to do manual scrolling for now ....
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

        for a in soup.find_all('a', href=True):
            if a.has_attr('class') and 'mega-item-header-link' in a['class']:
                #TODO: get time of the articles

                link = default_url + a['href']
                # check for duplicates
                if link not in url_list:
                    url_list.append(link)

        if lastCount == lenOfPage:
            match = True


    return url_list


#print(extract_text_from_url(url))

url_list = urls_from_domain('AAPL')
print(url_list)
print(len(url_list))


print(extract_text_from_url(url_list[0]))
