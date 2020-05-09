import data_collector
import pandas as pd
import stock_data
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
from nltk.tokenize import sent_tokenize
import random
import matplotlib.pyplot as plt

text = data_collector.load_data('apple', 'keyword')
#text = data_collector.title_from_url_list('apple')


#label = data_collector.load_data('AAPL', 'label')
label = stock_data.stock_price_label('AAPL', 14, 5) #'AAPL',14,5
#label = stock_data.stock_price_label_binary('AAPL', 7, '2010-01-01')
#label = stock_data.market_stock_label_binary('AAPL', 'IXIC', 7 , '2010-01-01')

featurelist = ['mentioned vs not mentioned',        # 회사 언급된 문장 수 / 언급되지 않은 문장 수
               'company mentioned polarity scores', # 회사 언급된 문장의 polarity score 평균
               'total polarity scores',             # (당일) 전체 기사의 polarity score
               'word related to increase',          # 기사에서 찾을 수 있는 increase 관련 단어 수
               'word related to decrease',          # 기사에서 찾을 수 있는 decrease 관련 단어 수
               'number of news']                    # (당일) 기사의 개수 (Need modification, not applicable yet)



def keyword_mentioned_sentence(txt, keyword):
    '''
    makes a list of sentences with the keyword mentioned.

    :param txt: word tokenized text
    :param keyword: targeting word.
    return: a list of sentences
    '''
    word_lowered = [w.lower() for w in txt]
    word_lowered_string = ' '.join(word_lowered)
    text_by_sentence =  sent_tokenize(word_lowered_string)
    filtered = [s for s in text_by_sentence if keyword in s]
    return filtered



def featurizer(text, company_name):
    '''


    :param text: text in the form of a list of words
    :param company_name: company name in string
    :return: featurized dictionary
    '''

    company_name = company_name.lower()
    increase = ['increase', 'up', 'rise', 'jump', 'rose', 'high', 'beating', 'positive']
    decrease = ['decrease', 'down', 'fall', 'plunge','low', 'negative']
    # increase, decrease word mentioned in specific sentences or the entire text
    feature = {'mentioned vs not mentioned':0,
               'company mentioned polarity scores':0,
               'total polarity scores': 0,
               'word related to increase': 0,
               'word related to decrease':0,
               'number of news': 0}   #'word related to increase': 0, 'word related to decrease':0
    sid = sia()
    mentioned_sentences = keyword_mentioned_sentence(text, company_name)
    mentioned_sentence_scores = [float(sid.polarity_scores(sent)['compound']) for sent in mentioned_sentences] # polarity score list for mentioned sentences.
    txt = [word.lower() for word in text]
    fd = nltk.FreqDist(txt)
    company_mentioned = fd[company_name]
    company_not_mentioned = len(sent_tokenize(' '.join(txt))) - company_mentioned
    increase_related_freq = [fd[word] for word in increase]
    decrease_related_freq = [fd[word] for word in decrease]

    feature['mentioned vs not mentioned'] = company_not_mentioned/company_mentioned
    feature['company mentioned polarity scores'] = sum(mentioned_sentence_scores) if len(mentioned_sentences) == 0\
                                                        else sum(mentioned_sentence_scores)/len(mentioned_sentences)
    feature['total polarity scores'] = float(sid.polarity_scores(' '.join(text))['compound'])
    feature['word related to increase'] = sum(increase_related_freq)
    feature['word related to decrease'] = sum(decrease_related_freq)

    return feature

#growth_rate = stock_data.market_stock_growth_interval('AAPL', 'IXIC', 14, '2010-01-01') # for feature vs rate
#merged_data = data_collector.merge_price_text(text, growth_rate) # for feature vs rate #(text, label)
merged_data = data_collector.merge_price_text(text, label)

'''아래 data가 training & testing에 이용될 수 있는 featureset!'''
featureset = [(featurizer(pair[0], 'apple'), tuple(pair[1])) for pair in merged_data] # data to train


#featureset = [(featurizer(pair[0], 'apple'), float(pair[1])) for pair in merged_data] # for feature vs rate
#news_number = data_collector.newsnumber_by_date('apple')
#merged_newsnumber_price = data_collector.merge_price_newsnumber(news_number, growth_rate) # for feature vs rate


def featureset_plotdata(feature_set):
    comp_mentioned = [feat[0]['mentioned vs not mentioned'] for feat in feature_set]
    comp_mentioned_ps = [feat[0]['company mentioned polarity scores'] for feat in feature_set]
    tot_ps = [feat[0]['total polarity scores'] for feat in feature_set]
    increase_related = [feat[0]['word related to increase'] for feat in feature_set]
    decrease_related = [feat[0]['word related to decrease'] for feat in feature_set]
    label_list = [int((-1)**(feat[1].index(1)/2)) if feat[1].index(1) != 1 else 0 for feat in feature_set]

    return comp_mentioned, comp_mentioned_ps, tot_ps, increase_related, decrease_related, label_list


def feature_vs_growth_rate(feature_set):
    comp_mentioned = [feat[0]['mentioned vs not mentioned'] for feat in feature_set]
    comp_mentioned_ps = [feat[0]['company mentioned polarity scores'] for feat in feature_set]
    tot_ps = [feat[0]['total polarity scores'] for feat in feature_set]
    increase_related = [feat[0]['word related to increase'] for feat in feature_set]
    decrease_related = [feat[0]['word related to decrease'] for feat in feature_set]
    y_list = [feat[1] for feat in feature_set]
    return comp_mentioned, comp_mentioned_ps, tot_ps, increase_related, decrease_related, y_list







## Featureset이 주가의 경향성을 어떻게 반영하는지

x1, x2, x3, x4, x5, y = featureset_plotdata(featureset)
#x1, x2, x3, x4, x5, y = feature_vs_growth_rate(featureset) # for feature vs rate


plt.subplot(1,5,1)
plt.title('Company Mentioned vs Label')
plt.scatter(x1, y)
plt.ylabel('increase or decrease')

plt.subplot(1,5,2)
plt.title('company mentioned polarity scores')
plt.scatter(x2, y)

plt.ylabel('increase or decrease')

plt.subplot(1,5,3)
plt.title('total polarity scores')
plt.scatter(x3, y)
plt.ylabel('increase or decrease')

plt.subplot(1,5,4)
plt.title('word related to increase')
plt.scatter(x4, y)
plt.ylabel('increase or decrease')

plt.subplot(1,5,5)
plt.title('word related to decrease')
plt.scatter(x5, y)
plt.ylabel('increase or decrease')

'''
plt.subplot(1,6,6)
plt.title('Number of news')
plt.scatter(x6, y)
plt.ylabel('increase or decrease')
'''
plt.show()
