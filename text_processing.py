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
label = stock_data.stock_price_label('AAPL', 14, 7) #'AAPL',14,5
#label = stock_data.stock_price_label_binary('AAPL', 7, '2010-01-01')
#label = stock_data.market_stock_label_binary('AAPL', 'IXIC', 7 , '2010-01-01')


def keyword_mentioned_sentence(txt, keyword):
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
    feature = {'company mentioned':0,
               'company mentioned polarity scores':0,
               'total polarity scores': 0,
               'word related to increase': 0,
               'word related to decrease':0}   #'word related to increase': 0, 'word related to decrease':0
    sid = sia()
    mentioned_sentences = keyword_mentioned_sentence(text, company_name)
    mentioned_sentence_scores = [float(sid.polarity_scores(sent)['compound']) for sent in mentioned_sentences] # polarity score list for mentioned sentences.
    txt = [word.lower() for word in text]
    fd = nltk.FreqDist(txt)
    company_mentioned = fd[company_name]
    increase_related_freq = [fd[word] for word in increase]
    decrease_related_freq = [fd[word] for word in decrease]

    feature['company mentioned'] = company_mentioned
    feature['company mentioned polarity scores'] = sum(mentioned_sentence_scores) if len(mentioned_sentences) == 0\
                                                        else sum(mentioned_sentence_scores)/len(mentioned_sentences)
    feature['total polarity scores'] = float(sid.polarity_scores(' '.join(text))['compound'])
    feature['word related to increase'] = sum(increase_related_freq)
    feature['word related to decrease'] = sum(decrease_related_freq)

    return feature


merged_data = data_collector.merge_price_text(text, label)
featureset = [(featurizer(pair[0], 'apple'), tuple(pair[1])) for pair in merged_data]
print(featureset)



#print(featureset[:50])
#random.shuffle(featureset)

#train_test_boundary = int(len(featureset)*0.8)
#train_set, test_set = featureset[:train_test_boundary], featureset[train_test_boundary:]

#classifier = nltk.DecisionTreeClassifier.train(train_set)


def featureset_plotdata(featureset):
    comp_mentioned = [feat[0]['company mentioned'] for feat in featureset]
    comp_mentioned_ps = [feat[0]['company mentioned polarity scores'] for feat in featureset]
    tot_ps = [feat[0]['total polarity scores'] for feat in featureset]
    increase_related = [feat[0]['word related to increase'] for feat in featureset]
    decrease_related = [feat[0]['word related to decrease'] for feat in featureset]
    label_list = [int((-1)**(feat[1].index(1)/2)) if feat[1].index(1) != 1 else 0 for feat in featureset]

    return comp_mentioned, comp_mentioned_ps, tot_ps, increase_related, decrease_related, label_list


x1, x2, x3, x4, x5, y = featureset_plotdata(featureset)


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

plt.show()



'''
classifier_input = data_collector.merge_price_text(filtered,label)
minus = 0
increase = 0
decrease = 0
maintain = 0
for i, pair in enumerate(classifier_input):
    #print('date: ', classifier_input.index[i], 'length: ', len(pair[0]), "sentences  News: ", ' '.join(pair[0]))
    print('label: ', pair[1], 'polarity score: ', sid.polarity_scores(' '.join(pair[0]))['compound'])
    if sid.polarity_scores(' '.join(pair[0]))['compound'] < 0:
        minus += 1
        if pair[1][0] == 1:
            increase += 1
        elif pair[1][1] == 1:
            maintain += 1
        else:
            decrease += 1

print('increase when minus: ', increase/minus, ' maintain when minus: ', maintain/minus, ' decrease when minus: ', decrease/minus)
print('total minus: ', minus)




stops = set(stopwords.words('english'))

result = [w for w in sample if w.lower() not in stops and w.lower() not in string.punctuation]
print(sample, '\n length: ', len(sample))

print(result, '\n length: ', len(result))
tresult = ' '.join(result)
print(nltk.FreqDist(result).most_common())
print(sid.polarity_scores(tresult))
print(sid.polarity_scores(' '.join(sample)))

#classifier_input = data_collector.merge_price_text(text, label)


sid = sia()

for pair in classifier_input[:100]:
    txt = ' '.join(pair[0])
    result = 0
    if pair[1] == [1,0,0]:
        result = 'increase'
    elif pair[1] == [0,1,0]:
        result = 'maintain'
    else:
        result = 'drop'
    if sid.polarity_scores(txt)['compound'] < 0:
        print(sid.polarity_scores(txt), ' ', result)
        print(txt)
        print(nltk.FreqDist(pair[0]).most_common())
'''