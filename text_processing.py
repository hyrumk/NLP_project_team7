import data_collector
import pandas as pd
import stock_data
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import random
import matplotlib.pyplot as plt

'''
Adjust the variable COMPANY. It affects line 23 (text), 247 (company name) featurizer part when you change the company
'''
COMPANY = 'Dow Jones'
TICKER = 'DJI'
'''
Loading a Text Data.
'''
# a list of news in a string form for each day.
text = data_collector.load_data('dowjones', 'gfkeyword')
#text = data_collector.load_data('apple_technews', 'keyword')
#text = data_collector.load_data(COMPANY.lower(), 'gfkeyword')
print(text)
#for txt in text:
#    if len(txt) > 1:
#        print(txt)
#text = data_collector.title_from_url_list('apple')

'''
Selecting a Type of Label
'''
#label = data_collector.load_data('AAPL', 'label')
#label = stock_data.stock_price_label('AAPL', interval = 3, percentage_rate = 1) # company mentioned polarity correlates.
#label = stock_data.stock_price_label2('AAPL', interval = 1, consecutive = 3)
#label = stock_data.stock_price_label3('AAPL', 'IXIC', 1, percentage_rate = 2, start_date = '2010-01-01')
#label = stock_data.stock_price_label_binary('AAPL', 10) #'AAPL',14,5
label = stock_data.stock_price_label_binary(TICKER.upper(), 1, '2009-06-01')
#label = stock_data.stock_price_label_binary2(TICKER.upper(), 'IXIC', 1 , '2010-01-01')
#label = stock_data.stock_price_label_binary3('AAPL', 'IXIC', 1 , '2010-01-01')



#growth_rate = stock_data.market_stock_growth_interval('AAPL', 'IXIC', 14, '2010-01-01') # for feature vs rate
#merged_data = data_collector.merge_price_text(text, growth_rate) # for feature vs rate #(text, label)
merged_data = data_collector.merge_price_text(text, label)



def addup_fd(total_fd, fd_to_add):
    '''
    This function is for the function 'extract_related_keyword'.

    :param total_fd: the fd to be added to.
    :param fd_to_add: the fd to add.
    :return: each element of the fd_to_add added to total_fd
    '''
    for word in fd_to_add:
        if word in total_fd:
            total_fd[word] += fd_to_add[word]
        else:
            total_fd[word] = fd_to_add[word]
    return total_fd


def extract_related_keyword(merged_seq, most_common = 50, ngram = 1):
    '''
    Find a dominantly appearing keywords when the stock is in rise and in fall.

    :param merged_seq: a pandas Series where each element is a list of news and a label.
    :param most_common: how many common words from FreqDist to first sort out
    :param ngram: which ngram we target.
    :return: keywords that appear dominantly for each label.
    '''
    increase = []
    decrease = []
    unnecessary = stopwords.words('english')
    unnecessary.append("’")
    total_fd_increased, total_fd_decreased = nltk.FreqDist({}), nltk.FreqDist({})
    increased_total, decreased_total = 0, 0
    for pair in merged_seq:
        news = pair[0]
        labeled = pair[1]
        if labeled == [1,0,0]:
            for txt in news:
                txt = txt.translate(str.maketrans('', '', string.punctuation))
                tokenized = word_tokenize(txt)
                filtered_tokenized = [word.lower() for word in tokenized if word.lower() not in unnecessary and not word.lower().isdecimal()]
                if ngram == 1:
                    fd = nltk.FreqDist(filtered_tokenized)
                elif ngram == 2:
                    fd = nltk.FreqDist(nltk.bigrams(filtered_tokenized))
                else: continue
                for word in fd:
                    fd[word] /= float(fd.N())
                total_fd_increased = addup_fd(total_fd_increased, fd)
                increased_total += 1
        elif labeled == [0,0,1]:
            for txt in news:
                txt = txt.translate(str.maketrans('', '', string.punctuation))
                tokenized = word_tokenize(txt)
                filtered_tokenized = [word.lower() for word in tokenized if word.lower() not in unnecessary and not word.lower().isdecimal()]
                if ngram == 1:
                    fd = nltk.FreqDist(filtered_tokenized)
                elif ngram == 2:
                    fd = nltk.FreqDist(nltk.bigrams(filtered_tokenized))
                else: continue
                for word in fd:
                    fd[word] /= float(fd.N())
                total_fd_decreased = addup_fd(total_fd_decreased, fd)
                decreased_total += 1
        else:
            continue
    for word in total_fd_increased:
        total_fd_increased[word] /= float(increased_total)
    for word in total_fd_decreased:
        total_fd_decreased[word] /= float(decreased_total)
    inc_list, dec_list = total_fd_increased.most_common(most_common), total_fd_decreased.most_common(most_common)
    #inc_list, dec_list = (total_fd_increased - total_fd_decreased).most_common(), (total_fd_decreased - total_fd_increased).most_common()
    inc_word_list = [pair[0] for pair in inc_list][:most_common]
    dec_word_list = [pair[0] for pair in dec_list][:most_common]
    inc_filtered = [pair for pair in inc_list if pair[0] not in dec_word_list]
    dec_filtered = [pair for pair in dec_list if pair[0] not in inc_word_list]
    # inc dec 각각 제일 흔한 것 50개 잡고 겹치지 않도록 겹치는 것들은 전부 제거, 그리고 남은 안 겹치는 것들 중에 freq별로 정리?
    return inc_filtered, dec_filtered # lists of high frequency words in increase and decrease [(word, frequency) *]


inc, dec = extract_related_keyword(merged_data)
inc_bi, dec_bi = extract_related_keyword(merged_data, ngram = 2)
inc_list = [pair[0] for pair in inc]
dec_list = [pair[0] for pair in dec]
inc_bi_list = [' '.join(list(pair[0])) for pair in inc_bi]
dec_bi_list = [' '.join(list(pair[0])) for pair in dec_bi]
print(inc_list)
print(dec_list)
print(inc_bi_list)
print(dec_bi_list)


'''아래 data가 training & testing에 이용될 수 있는 featureset!'''


featurelist = ['company mentioned polarity scores', # 회사 언급된 문장의 polarity score 평균
               'word related to increase',          # 기사에서 찾을 수 있는 increase 관련 단어 수
               'word related to decrease',          # 기사에서 찾을 수 있는 decrease 관련 단어 수
               'monogram_inc', 'monogram_dec',
               'bigram_inc', 'bigram_dec']                    # (당일) 기사의 개수 (Need modification, not applicable yet)



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





def featurizer(texts, company_name):
    '''


    :param text: text in the form of a list of strings
    :param company_name: company name in string
    :return: featurized dictionary
    '''
    text = []
    news_count = 0
    for news in texts:
        news_count += 1
        text += word_tokenize(news)
    company_name = company_name.lower()
    increase = {'increase', 'up', 'rise', 'jump', 'rose', 'high', 'beating', 'positive', 'gained', 'climbed',
                'jumped', 'surged', 'rising', 'increased', 'soared', 'surging', 'skyrocketed', 'climb', 'climbing',
                'gains', 'surge', 'grew', 'jumping'
                }
    decrease = {'decrease', 'down', 'fall', 'plunge','low', 'negative', 'fell', 'lost', 'dropped', 'declined',
                'tumbled', 'slipped', 'slumped', 'dipped', 'plunged', 'falling', 'slid', 'plummeted', 'sank',
                'decline', 'dropping', 'tumbling'
                }
    # increase, decrease word mentioned in specific sentences or the entire text
    feature = {'company mentioned polarity scores':0,
               'word related to increase': 0,
               'word related to decrease':0,
               'monogram_inc': 0, 'monogram_dec': 0,
              'bigram_inc': 0, 'bigram_dec': 0
                }   #'word related to increase': 0, 'word related to decrease':0
    sid = sia()
    mentioned_sentences = keyword_mentioned_sentence(text, company_name)
    mentioned_sentence_scores = [float(sid.polarity_scores(sent)['compound']) for sent in mentioned_sentences] # polarity score list for mentioned sentences.
    txt = [word.lower() for word in text]
    fd = nltk.FreqDist(txt)
    company_mentioned = fd[company_name]
    #company_not_mentioned = len(sent_tokenize(' '.join(txt))) - company_mentioned
    increase_related_freq = [fd[word] for word in increase]
    decrease_related_freq = [fd[word] for word in decrease]
    increase_keyword, decrease_keyword = inc_list, dec_list
    key_inc, key_dec = [], []
    for word in inc_list:
        key_inc += keyword_mentioned_sentence(txt, word)
    for word in dec_list:
        key_dec += keyword_mentioned_sentence(txt, word)
    increase_keyword_scores = [float(sid.polarity_scores(sent)['compound']) for sent in key_inc]
    decrease_keyword_scores = [float(sid.polarity_scores(sent)['compound']) for sent in key_dec]

    key_inc_bigram, key_dec_bigram = [], []
    for word in inc_bi_list:
        key_inc_bigram += keyword_mentioned_sentence(txt, word)
    for word in dec_bi_list:
        key_dec_bigram += keyword_mentioned_sentence(txt, word)
    increase_bigram_keyword_scores = [float(sid.polarity_scores(sent)['compound']) for sent in key_inc_bigram]
    decrease_bigram_keyword_scores = [float(sid.polarity_scores(sent)['compound']) for sent in key_dec_bigram]

    #feature['mentioned vs not mentioned'] = 1#company_mentioned/company_not_mentioned
    feature['company mentioned polarity scores'] = sum(mentioned_sentence_scores) if len(mentioned_sentences) == 0\
                                                        else sum(mentioned_sentence_scores)/len(mentioned_sentences)
    #feature['total polarity scores'] = float(sid.polarity_scores(' '.join(text))['compound'])
    feature['word related to increase'] = sum(increase_related_freq)/len(txt) if len(text) != 0\
                                                                            else sum(increase_related_freq)#sum([fd[word]/fd.N() for word in increase_keyword])
    feature['word related to decrease'] = sum(decrease_related_freq)/len(txt)if len(text) != 0\
                                                                            else sum(decrease_related_freq)#sum([fd[word]/fd.N() for word in decrease_keyword])
    #feature['number of news'] = news_count
    feature['monogram_inc'] = sum(increase_keyword_scores) if len(key_inc) == 0\
                                                        else sum(increase_keyword_scores)/len(key_inc)
    feature['monogram_dec'] = sum(decrease_keyword_scores) if len(key_dec) == 0 \
        else sum(decrease_keyword_scores) / len(key_dec)
    feature['bigram_inc'] = sum(increase_bigram_keyword_scores) if len(key_inc_bigram) == 0 \
        else sum(increase_bigram_keyword_scores) / len(key_inc_bigram)
    feature['bigram_dec'] = sum(decrease_bigram_keyword_scores) if len(key_dec_bigram) == 0 \
        else sum(decrease_bigram_keyword_scores) / len(key_dec_bigram)

    return feature



#featureset = [(featurizer(pair[0], COMPANY.lower()), tuple(pair[1])) for pair in merged_data] # data to train

featureset = [(featurizer(pair[0], COMPANY.lower()), tuple(pair[1])) for pair in merged_data] # for normal label
#featureset = [(featurizer(pair[0], COMPANY.lower()), pair[1]) for pair in merged_data] # for feature vs rate


def featureset_plotdata(feature_set):
    #comp_mentioned = [feat[0]['mentioned vs not mentioned'] for feat in feature_set]
    comp_mentioned_ps = [feat[0]['company mentioned polarity scores'] for feat in feature_set]
    #tot_ps = [feat[0]['total polarity scores'] for feat in feature_set]
    increase_related = [feat[0]['word related to increase'] for feat in feature_set]
    decrease_related = [feat[0]['word related to decrease'] for feat in feature_set]
    #news_number = [feat[0]['number of news'] for feat in feature_set]
    monogram_inc_keyword = [feat[0]['monogram_inc'] for feat in feature_set]
    monogram_dec_keyword = [feat[0]['monogram_dec'] for feat in feature_set]
    bigram_inc_keyword = [feat[0]['bigram_inc'] for feat in feature_set]
    bigram_dec_keyword = [feat[0]['bigram_dec'] for feat in feature_set]
    label_list = [int((-1)**(feat[1].index(1)/2)) if feat[1].index(1) != 1 else 0 for feat in feature_set]

    return comp_mentioned_ps, increase_related, decrease_related,\
            monogram_inc_keyword, monogram_dec_keyword, bigram_inc_keyword, bigram_dec_keyword, label_list


def feature_vs_growth_rate(feature_set):
    #comp_mentioned = [feat[0]['mentioned vs not mentioned'] for feat in feature_set]
    comp_mentioned_ps = [feat[0]['company mentioned polarity scores'] for feat in feature_set]
    tot_ps = [feat[0]['total polarity scores'] for feat in feature_set]
    increase_related = [feat[0]['word related to increase'] for feat in feature_set]
    decrease_related = [feat[0]['word related to decrease'] for feat in feature_set]
    news_number = [feat[0]['number of news'] for feat in feature_set]
    monogram_inc_keyword = [feat[0]['monogram_inc'] for feat in feature_set]
    monogram_dec_keyword = [feat[0]['monogram_dec'] for feat in feature_set]
    bigram_inc_keyword = [feat[0]['bigram_inc'] for feat in feature_set]
    bigram_dec_keyword = [feat[0]['bigram_dec'] for feat in feature_set]
    y_list = [feat[1] for feat in feature_set]
    return comp_mentioned_ps, tot_ps, increase_related, decrease_related, news_number,\
            monogram_inc_keyword, monogram_dec_keyword, bigram_inc_keyword, bigram_dec_keyword, y_list






## Featureset이 주가의 경향성을 어떻게 반영하는지

x1, x2, x3, x4, x5, x6, x7, y = featureset_plotdata(featureset)
#x2, x3, x4, x5, x6, x7, x8, x9, x10, y = feature_vs_growth_rate(featureset) # for feature vs rate

'''
plt.subplot(1,8,1)
plt.title('Company Mentioned vs Label')
plt.scatter(x1, y)
plt.ylabel('increase or decrease')
'''
plt.subplot(1,7,1)
plt.title('company mentioned polarity scores')
plt.scatter(y, x1)
plt.ylabel('increase or decrease')

plt.subplot(1,7,2)
plt.title('Increase mentioned')
plt.scatter(y, x2)
plt.ylabel('increase or decrease')

plt.subplot(1,7,3)
plt.title('Decrease mentioned')
plt.scatter(y, x3)
plt.ylabel('increase or decrease')

plt.subplot(1,7,4)
plt.title('Monogram inc polarity')
plt.scatter(y, x4)
plt.ylabel('increase or decrease')


plt.subplot(1,7,5)
plt.title('Monogram dec polarity')
plt.scatter(y, x5)
plt.ylabel('increase or decrease')

plt.subplot(1,7,6)
plt.title('Bigram inc polarity')
plt.scatter(y, x6)
plt.ylabel('increase or decrease')

plt.subplot(1,7,7)
plt.title('Bigram dec polarity')
plt.scatter(y, x7)
plt.ylabel('increase or decrease')

plt.suptitle(COMPANY)

plt.show()







##################################################################################################
def find_keyword(merged_data):
    increase_keyword = []
    decrease_keyword = []
    maintain_keyword = []
    for pair in merged_data:
        newslist = pair[0]
        label = pair[1]
        total_news = []
        for news in newslist:
            single_news = word_tokenize(news)
            total_news += single_news
        nltk.FreqDist(total_news)