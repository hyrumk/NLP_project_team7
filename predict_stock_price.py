import re
import time
import random
import pandas as pd
import datetime as dt
from collections import defaultdict

import nltk
from nltk.classify import maxent
from nltk.metrics.scores import f_measure
from nltk import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia

import stock_data
import data_collector
from text_processing import keyword_mentioned_sentence


with open('apples.txt', 'r', encoding='utf-8-sig') as f:
    new_data = f.readlines()

new_index = []
new_text = []
con = False
temp = ''
for line in new_data:
    if not re.sub(r'\d\d\d\d[-]\d\d[-][\d]?\d', '', line).strip():
        new_text.append(temp.strip())
        temp = ''
        year, month, date = line.strip().split('-')
        if len(date)==1:
            date = '0' + date
        index = dt.datetime(int(year), int(month), int(date))
        new_index.append(index)
    elif not re.sub(r'\d', '', line).strip():
        num = int(line.strip())
        if num:
            con = True
        else:
            con = False
    elif con:
        temp += ' ' + line.strip()
new_text.append(temp.strip())     
new_text = new_text[1:]
new_index2 = []
new_text2 = []
for i, text in enumerate(new_text):
    if text:
        new_text2.append(text)
        new_index2.append(new_index[i])
new_text2 = [[article] for article in new_text2]

new_text = pd.Series(new_text2, index=new_index2).sort_index()
label = stock_data.stock_price_label3('AAPL', 'IXIC', 1, percentage_rate = 3, start_date = '2010-01-01')
merged_data = data_collector.merge_price_text(new_text, label)

def featurizer(texts, company_name='apple'):
    '''
    :param text: text in the form of a list of strings
    :param company_name: company name in string
    :return: featurized dictionary
    '''
    text = []
    for news in texts:
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
    increase_related_freq = [fd[word]/fd.N() for word in increase]
    decrease_related_freq = [fd[word]/fd.N() for word in decrease]

    feature['mentioned vs not mentioned'] = 1#company_mentioned/company_not_mentioned
    feature['company mentioned polarity scores'] = sum(mentioned_sentence_scores) if len(mentioned_sentences) == 0\
                                                        else sum(mentioned_sentence_scores)/len(mentioned_sentences)
    feature['total polarity scores'] = float(sid.polarity_scores(' '.join(text))['compound'])
    feature['word related to increase'] = sum(increase_related_freq)
    feature['word related to decrease'] = sum(decrease_related_freq)

    return feature


feature_set = [(featurizer(article, company_name='apple'), tuple(l)) for article, l in merged_data
               if article]

if __name__=='__main__':
    random.shuffle(feature_set)
    slicing_point = int(len(feature_set) * 0.1)
    train_set, test_set = feature_set[slicing_point:], feature_set[:slicing_point]
    start = time.time()
    classifier = maxent.MaxentClassifier.train(train_set, 'gis')
    print('training time:', time.time()-start)
    test = [feature for feature, _ in test_set]
    labels = [label for _, label in test_set]
    refsets, testsets = defaultdict(set), defaultdict(set)
    for i, (feats, label) in enumerate(test_set):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
    rise, maintain, fall = [], [], []
    for i in range(len(test)):
            pdist = classifier.prob_classify(test[i])
            guess = classifier.classify(test[i])
            gold = labels[i]
            if gold==(1, 0, 0):
                rise.append((guess, pdist.prob((1, 0, 0)), pdist.prob((0, 1, 0)), pdist.prob((0, 0, 1))))
            elif gold==(0, 1, 0):
                maintain.append((guess, pdist.prob((1, 0, 0)), pdist.prob((0, 1, 0)), pdist.prob((0, 0, 1))))
            elif gold==(0, 0, 1):
                fall.append((guess, pdist.prob((1, 0, 0)), pdist.prob((0, 1, 0)), pdist.prob((0, 0, 1))))                
            print(pdist.prob((1, 0, 0)), pdist.prob((0, 1, 0)), pdist.prob((0, 0, 1)), gold)
    print('accuracy:', nltk.classify.accuracy(classifier, test_set))
    print('f-measure (1, 0, 0):', f_measure(refsets[(1, 0, 0)], testsets[(1, 0, 0)]))
    print('f-measure (0, 1, 0):', f_measure(refsets[(0, 1, 0)], testsets[(0, 1, 0)]))
    print('f-measure (0, 0, 1):', f_measure(refsets[(0, 0, 1)], testsets[(0, 0, 1)]))    
    print()
    print('rise')
    for i in rise:
        print(i)
    print('fall')
    for i in fall:
        print(i)

    import numpy
    import matplotlib.pyplot as plt
    company_polarity = [f['company mentioned polarity scores'] for f, l in feature_set]
    company_polarityR = [f['company mentioned polarity scores'] for f, l in feature_set
                         if l==(1, 0, 0)]
    company_polarityM = [f['company mentioned polarity scores'] for f, l in feature_set
                         if l==(0, 1, 0)]
    company_polarityF = [f['company mentioned polarity scores'] for f, l in feature_set
                         if l==(0, 0, 1)]
    print('\ncompany mentioned polarity scores')
    print('mean:', numpy.mean(company_polarityR), 'std:', numpy.std(company_polarityR), 'median:', numpy.median(company_polarityR))
    print('mean:', numpy.mean(company_polarityM), 'std:', numpy.std(company_polarityM), 'median:', numpy.median(company_polarityM))
    print('mean:', numpy.mean(company_polarityF), 'std:', numpy.std(company_polarityF), 'median:', numpy.median(company_polarityF))

    total_polarity =  [f['total polarity scores'] for f, l in feature_set]
    total_polarityR =  [f['total polarity scores'] for f, l in feature_set
                       if l==(1, 0, 0)]
    total_polarityM =  [f['total polarity scores'] for f, l in feature_set
                       if l==(0, 1, 0)]
    total_polarityF =  [f['total polarity scores'] for f, l in feature_set
                       if l==(0, 0, 1)]
    print('\ntotal polarity scores')
    print('mean:', numpy.mean(total_polarityR), 'std:', numpy.std(total_polarityR), 'median:', numpy.median(total_polarityR))
    print('mean:', numpy.mean(total_polarityM), 'std:', numpy.std(total_polarityM), 'median:', numpy.median(total_polarityM))
    print('mean:', numpy.mean(total_polarityF), 'std:', numpy.std(total_polarityF), 'median:', numpy.median(total_polarityF))

    increase = [f['word related to increase'] for f, l in feature_set]
    increaseR = [f['word related to increase'] for f, l in feature_set
                 if l==(1, 0, 0)]
    increaseM = [f['word related to increase'] for f, l in feature_set
                 if l==(0, 1, 0)]
    increaseF = [f['word related to increase'] for f, l in feature_set
                 if l==(0, 0, 1)]
    print('\nword related to increase')
    print('mean:', numpy.mean(increaseR), 'std:', numpy.std(increaseR), 'median:', numpy.median(increaseR))
    print('mean:', numpy.mean(increaseM), 'std:', numpy.std(increaseM), 'median:', numpy.median(increaseM))
    print('mean:', numpy.mean(increaseF), 'std:', numpy.std(increaseF), 'median:', numpy.median(increaseF))

    decrease = [f['word related to decrease'] for f, l in feature_set]
    decreaseR = [f['word related to decrease'] for f, l in feature_set
                 if l==(1, 0, 0)]
    decreaseM = [f['word related to decrease'] for f, l in feature_set
                 if l==(0, 1, 0)]
    decreaseF = [f['word related to decrease'] for f, l in feature_set
                 if l==(0, 0, 1)]
    print('\nword related to decrease')
    print('mean:', numpy.mean(decreaseR), 'std:', numpy.std(decreaseR), 'median:', numpy.median(decreaseR))
    print('mean:', numpy.mean(decreaseM), 'std:', numpy.std(decreaseM), 'median:', numpy.median(decreaseM))
    print('mean:', numpy.mean(decreaseF), 'std:', numpy.std(decreaseF), 'median:', numpy.median(decreaseF))













    





