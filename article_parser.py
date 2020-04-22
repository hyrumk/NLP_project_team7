import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests


url = 'http://www.prnewswire.com/news-releases/tata-consultancy-services-reports-broad-based-growth-across-markets-marks-steady-fy17-300440934.html'

def extract_text_from_url(url):
    '''
    Extracts useful texts from provided url
    Step 1.
        Remove texts under irrelevant parent html tags

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

    print(output)

