import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests

from nltk.tokenize import word_tokenize

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import string
import time

# PIP INSTALLS: selenium, bs4


import json

class GimmeProxyAPI(object):
	"""docstring for proxy"""
	def __init__(self, **args):
		self.base_url = "https://gimmeproxy.com/api/getProxy"
		self.response = None

		if self.response is None:
			self.response = self.get_proxy(args=args)
   
	def get_proxy(self, **args):

		request = requests.get(self.base_url, params=args)

		if request.status_code == 200:
			self.response = request.json()
		else:
			raise Exception("An unknown error occured, status_code = {}".format(request.status_code))

		return self.response


	def get_curl(self):
		curl = self.response["curl"]
		return curl

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
    
    headers = requests.utils.default_headers()
    headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
    })
    output = ''
    res = requests.get(url, headers = headers)
    soup = BeautifulSoup(res.content,"html.parser")
            
    headerDiv = soup.find('h1', {"class":"articleHeader"})
    mainDiv = soup.find('div', {"class":"articlePage"})
    output += headerDiv.text
    output += mainDiv.text


    
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



def urls_search_by_keyword(keyword):
    '''
    Makes a text file with a list of urls of a given keyword.

    :param keyword: keyword string
    :return: None
    '''

    url_list = urls_from_domain(keyword)
    keyword_name = keyword.translate({ord(c): None for c in string.whitespace})
    txt_file_name = './Data Storage/listfile_{}.txt'.format(keyword_name.lower())
    with open(txt_file_name, 'w') as filehandle:
        filehandle.truncate(0)
        for listitem in url_list:
            filehandle.write('%s , %s , %s\n' % (listitem[0], listitem[1], listitem[2]))


