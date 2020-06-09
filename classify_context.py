import nltk
from nltk.classify import maxent
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer

import stock_data
import data_collector

import os
import re
import pickle
import random
import itertools

stopwords = stopwords.words('english')
stemmer = LancasterStemmer()
texts = data_collector.load_data('apple', 'keyword')
labels = stock_data.stock_price_label('AAPL', 14, 5)
inputs = data_collector.merge_price_text(texts, labels).values # 233

articles = list(itertools.chain(*[article_list for article_list, _ in inputs])) # 2188

if os.path.exists('titles.pkl'):
    with open('titles.pkl', 'rb') as f:
        titles = pickle.load(f)
else:
    titles = []
    for article in articles:
        try:
            title = re.findall(r'.*?Investing[.]com|.*?Reuters', article)[0]
        except IndexError:
            title = nltk.sent_tokenize(article)[0]
        titles.append(title)

    with open('titles.pkl', 'wb') as f:
        pickle.dump(titles, f)


percents = [article.count('%')/len(nltk.word_tokenize(article)) for article in articles]
# mean = 0.02569

# developing with random 100 data
random_pick = [50, 52, 104, 107, 128, 144, 170, 173, 182, 192, 220, 221, 325, 328, 344, 348, 435, 443, 490, 507, 593, 628, 643, 667, 683, 688, 697, 718, 755, 792, 841, 851, 873, 878, 879, 907, 920, 939, 944, 952, 979, 993, 1014, 1055, 1064, 1114, 1121, 1122, 1136, 1205, 1208, 1247, 1248, 1265, 1282, 1284, 1361, 1372, 1381, 1404, 1435, 1457, 1461, 1478, 1508, 1516, 1535, 1580, 1588, 1610, 1637, 1677, 1678, 1710, 1736, 1747, 1761, 1803, 1810, 1823, 1830, 1848, 1863, 1876, 1877, 1878, 1960, 1976, 1984, 2010, 2018, 2023, 2031, 2055, 2061, 2069, 2093, 2104, 2179, 2186]
for i in random_pick:
    # print(i, titles[i], percents[i])
    pass

gold = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
gold += [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1]
gold += [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1]
gold += [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
gold += [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0]
gold_dict = {random_pick[i]: gold[i]>0 for i in range(len(random_pick))}

related = []
not_related = []
for i, label in enumerate(gold):
    if label:
        related.append(random_pick[i])
    else:
        not_related.append(random_pick[i])

'''
ratio of %

mean of total(random 100 picks) = 0.02545
(for all 2188 articles - 0.02569)

number of related = 91
mean of related = 0.02750
minimum of related = 0.0

number of not_related = 9
mean of not_related = 0.004726
maximum of not_related = 0.02299
'''

keywords = {'nasdaq', 'nyse', 'dow', 'stock', 'futures'}
c = []
con = True
for i in random_pick:
    for key in keywords:
        if key in titles[i].lower() and con:
            c.append(1)
            con = False
    if con:
        c.append(0)
    con = True

a = []
con = True
for i in related:
    for key in keywords:
        if key in titles[i].lower() and con:
            a.append(1)
            con = False
    if con:
        a.append(0)
    con = True

b = []
con = True
for i in not_related:
    for key in keywords:
        if key in titles[i].lower() and con:
            b.append(1)
            con = False
    if con:
        b.append(0)
    con = True

cc = []
for i in random_pick:
    temp = 0
    for key in keywords:
        temp += articles[i].lower().count(key)
    cc.append(temp/len(nltk.word_tokenize(articles[i])))

aa = []
for i in related:
    temp = 0
    for key in keywords:
        temp += articles[i].lower().count(key)
    aa.append(temp/len(nltk.word_tokenize(articles[i])))

bb = []
for i in not_related:
    temp = 0
    for key in keywords:
        temp += articles[i].lower().count(key)
    bb.append(temp/len(nltk.word_tokenize(articles[i])))

'''
existence of keywords in title

mean of total = 0.74
(for all 2188 articles - 0.7308)

number of related = 91
mean of related = 0.8132

number of not_related = 9
mean of not_related = 0.0


ratio of keywords in article

mean of total = 0.03027
(for all 2188 articles - 0.02945)

number of related = 91
mean of related = 0.03219
minimum of related = 0.0

number of not_related = 9
mean of not_related = 0.01082
maximum of not_related = 0.04789
'''

most_common = [('rose', 2210), ('fell', 1805), ('points', 1377), ('gained', 1034), ('added', 804), ('climbed', 658), ('jumped', 638), ('lost', 550), ('dropped', 550), ('declined', 532), ('surged', 463), ('tumbled', 427), ('jones', 421), ('advanced', 411), ('slipped', 362), ('edged', 330), ('inched', 316), ('shares', 307), ('rallied', 302), ('slumped', 273), ('average', 262), ('dipped', 257), ('indicated', 254), ('signaled', 251), ('plunged', 245), ('falling', 224), ('nearly', 217), ('pointed', 213), ('retreated', 206), ('rising', 195), ('slid', 195), ('eased', 184), ('aapl', 178), ('increased', 174), ('soared', 168), ('closed', 167), ('plummeted', 161), ('rise', 147), ('gaining', 145), ('shed', 131), ('futures', 129), ('traded', 128), ('expectations', 128), ('delivery', 127), ('gain', 123), ('options', 123), ('dow', 123), ('eur/usd', 122), ('losing', 121), ('index', 118), ('lows', 111), ('around', 110), ('decreased', 106), ('still', 77), ('rate', 77), ('back', 73), ('drop', 72), ('cents', 69), ('much', 67), ('500', 66), ('increase', 64), ('dax', 64), ('sank', 64), ('surging', 63), ('roughly', 62), ('highs', 62), ('decline', 61), ('finished', 61), ("'s", 54), ('skyrocketed', 52), ('currencies', 52), ('apple', 50), ('dove', 50), ('reported', 49), ('fall', 49), ('climb', 48), ('stock', 48), ('jump', 46), ('climbing', 46), ('nasdaq', 45), ('gains', 45), ('surge', 43), ('almost', 42), ('point', 42), ('dropping', 40), ('higher', 40), ('ba', 39), ('low', 39), ('unchanged', 38), ('grew', 37), ('growth', 37), ('approximately', 37), ('tumbling', 36), ('ticked', 35), ('revised', 35), ('gs', 34), ('jumping', 33), ('intc', 33), ('nyse', 32), ('adjusted', 32)]
most_common_vers = ['rose', 'fell', 'points', 'gained', 'added', 'climbed', 'jumped', 'lost', 'dropped', 'declined', 'surged', 'tumbled', 'jones', 'advanced', 'slipped', 'edged', 'inched', 'shares', 'rallied', 'slumped', 'average', 'dipped', 'indicated', 'signaled', 'plunged', 'falling', 'nearly', 'pointed', 'retreated', 'rising', 'slid', 'eased', 'aapl', 'increased', 'soared', 'closed', 'plummeted', 'rise', 'gaining', 'shed', 'futures', 'traded', 'expectations', 'delivery', 'gain', 'options', 'dow', 'eur/usd', 'losing', 'index', 'lows', 'around', 'decreased', 'still', 'rate', 'back', 'drop', 'cents', 'much', '500', 'increase', 'dax', 'sank', 'surging', 'roughly', 'highs', 'decline', 'finished', "'s", 'skyrocketed', 'currencies', 'apple', 'dove', 'reported', 'fall', 'climb', 'stock', 'jump', 'climbing', 'nasdaq', 'gains', 'surge', 'almost', 'point', 'dropping', 'higher', 'ba', 'low', 'unchanged', 'grew', 'growth', 'approximately', 'tumbling', 'ticked', 'revised', 'gs', 'jumping', 'intc', 'nyse', 'adjusted']
keyverbs = ['rose', 'fell', 'gained', 'added', 'climbed', 'jumped', 'lost', 'dropped', 'declined', 'surged', 'tumbled', 'advanced', 'slipped', 'edged', 'inched', 'rallied', 'slumped', 'dipped', 'indicated', 'signaled', 'plunged', 'falling', 'pointed', 'retreated', 'rising', 'slid', 'eased', 'increased', 'soared', 'closed', 'plummeted', 'rise', 'gaining', 'shed', 'futures', 'traded', 'gain', 'losing', 'decreased', 'drop', 'increase', 'sank', 'surging', 'roughly', 'highs', 'decline', 'finished', 'skyrocketed', 'fall', 'climb', 'stock', 'jump', 'climbing', 'gains', 'surge', 'point', 'dropping', 'higher', 'unchanged', 'grew', 'growth', 'tumbling', 'ticked', 'revised', 'jumping', 'adjusted']

ccc = []
for i in random_pick:
    temp = 0
    for key in keyverbs:
        temp += articles[i].lower().count(key)
    ccc.append(temp/len(nltk.word_tokenize(articles[i])))

aaa = []
for i in related:
    temp = 0
    for key in keyverbs:
        temp += articles[i].lower().count(key)
    aaa.append(temp/len(nltk.word_tokenize(articles[i])))

bbb = []
for i in not_related:
    temp = 0
    for key in keyverbs:
        temp += articles[i].lower().count(key)
    bbb.append(temp/len(nltk.word_tokenize(articles[i])))


'''
ratio of keyverbs in article

mean of total = 0.05686
(for all 2188 articles - 0.05666)

number of related = 91
mean of related = 0.06093
minimum of related = 0.0

number of not_related = 9
mean of not_related = 0.01570
maximum of not_related = 0.04225
'''

def features(article_index):
    article = articles[article_index].lower()
    title = titles[article_index].lower()
    words = nltk.word_tokenize(article)
    
    ratio_per_article = percents[article_index]

    exist_keyword_title = 0
    for key in keywords:
        if key in title:
            exist_keyword_title = 1
            break
        
    temp = 0
    for key in keywords:
        temp += article.count(key)
    ratio_keyword_article = temp/len(words)
   
    temp = 0
    for key in keyverbs:
        temp += article.count(key)
    ratio_keyverb_article = temp/len(words)

    return {'ratio_per_article': ratio_per_article, 'exist_keyword_title': exist_keyword_title, 'ratio_keyword_article': ratio_keyword_article, 'ratio_keyverb_article': ratio_keyverb_article}
"""   
test_index = [170, 173, 325, 490, 667, 920, 952, 979, 1055, 1284, 1588, 1610, 1761, 1810, 1848, 1863, 1877, 1976, 2010, 2069]
train_index = sorted(list(set(random_pick) - set(test_index)))
train_set = [(features(index), gold_dict[index]) for index in train_index]
test_set = [(features(index), gold_dict[index]) for index in test_index]
classifier = maxent.MaxentClassifier.train(train_set, 'gis', trace = 0)

for feature, gold in test_set:
    pdist = classifier.prob_classify(feature)
    print(pdist.prob(1), pdist.prob(0), gold)
"""
train_index = random_pick # 100
test_index = sorted(list(set(range(2188)) - set(train_index))) # 2088
train_set = [(features(index), gold_dict[index]) for index in train_index]
test = [features(index) for index in test_index]
classifier = maxent.MaxentClassifier.train(train_set, 'gis')

for i in range(10):
    n = random.choice(range(2088))
    pdist = classifier.prob_classify(test[n])
    print(titles[test_index[n]])
    print(pdist.prob(True), pdist.prob(False))

nl = []
for i in range(2088):
    pdist = classifier.prob_classify(test[i])
    if pdist.prob(True) < pdist.prob(False):
        nl.append(test_index[i])
        # print(titles[test_index[i]])
        # print(pdist.prob(True), pdist.prob(False))

# Ture : False = 1933 : 155 = 12.47 : 1








