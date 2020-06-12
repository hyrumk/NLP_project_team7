from serpapi.google_search_results import GoogleSearchResults
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
from collections import defaultdict
from nltk.tokenize import word_tokenize

def search(keyword, date_start, date_end):
	"""
	Function to find specific search result from Google.
	Using the already collated article data to seek wider breadth of information
	for the specific company at a specific time range

	Result from GoogleSearchResults returns a dictionary with the following fields in order:

	1. search_metadata
		- Includes "created_at" field useful for filtering results of specific date time
	2. search_parameters
	3. search_information
	4. organic_results
		- most relevant field
		- use "title" and "displayed_link" for extracting text data
	"""

	client = GoogleSearchResults({"q": keyword, "location": "Austin,Texas", "api_key": "demo", "created_at": "2019-04-05"})
	result = client.get_dict()

	return result




def search_from_tech_archive(year, month, ticker):
	"""
	month/day format in list: <li class="list-title date-heading"> text

	all the needed data under <ul class="basic-list"> in the form of list item

	"""
	domain_frame = 'https://www.techradar.com/news/archive/{year}/{month}'.format(year = year, month= month)


	headers = requests.utils.default_headers()
	headers.update({
    	'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
    })
    

	link_by_date = defaultdict(list)


	res = requests.get(domain_frame, headers = headers)
	soup = BeautifulSoup(res.content,"html.parser")
            
	unordered_list = soup.find('ul', {"class":"basic-list"})
	list_children = unordered_list.findChildren('li' , recursive=False)


	date_format = ""
	for list_item in list_children:


		if list_item.has_attr('class') and list_item['class'][0] == 'list-title':
			# its the title list
			day = word_tokenize(list_item.text)[-1]
			date_format = year + "-" + month + "-" + day
			link_by_date[date_format] = []


		else:
			#its the article list
			tmp_ul = list_item.findChildren('ul', {'class': 'day-list'})
			articles = tmp_ul[0].findChildren('li', {'class': 'day-article'})
			for article in articles:
				article_tokens = word_tokenize(article.text)
				if ticker in article_tokens:
					article_link = article.findChildren('a')[0]['href']
					link_by_date[date_format].append(article_link)




	return link_by_date


def tokenize_article_from_url(dictionary):
	"""
	From given url, aggregate text from <p> tags under <div> of class ="text-copy bodyCopy auto"
	"""

	for key in dictionary.keys():
		for i in range(len(dictionary[key])):
			url = dictionary[key][i]
			headers = requests.utils.default_headers()
			headers.update({
		    	'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
		    })

			res = requests.get(url, headers = headers)
			soup = BeautifulSoup(res.content,"html.parser")
			article_body = soup.find('div', {"class":"text-copy bodyCopy auto"})
			article_body_parts = article_body.findChildren('p')

			article_tokens = []
			for parts in article_body_parts:
				article_tokens += word_tokenize(parts.text)

			#print(article_tokens)
			dictionary[key][i] = article_tokens


	return dictionary



def get_article(year, month, day):
	date_format = year + "-" + month + "-" + day
	date_url_dict = search_from_tech_archive(year, month, 'Apple')
	date_token_dict = tokenize_article_from_url(date_url_dict)

	if len(date_token_dict[date_format]) != 0:
		return date_token_dict[date_format]
	else:
		return 0



date_url_dict = search_from_tech_archive('2010', '08', 'Apple')
date_token_dict = tokenize_article_from_url(date_url_dict)
print(get_article('2010', '08', '16'))









