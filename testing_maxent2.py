import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
from nltk.classify import maxent
import re
import time
import random
import itertools
import data_collector
import stock_data
import text_processing

stopwords = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def is_num(word):
    '''
    Check wether the given word means numeric or not.

    :param word: A word
    :type word: String
    :return True if word means numeric, False if not
    :rtype: Bool
    '''
    word = re.sub(r'^\$|\D$', '', word)
    try:
        float(word)
        return True
    except ValueError:
        return False

def normalizing(words):
    '''
    Get a list of words and normalizing them.
    
    Process of normalizing:
    1. Remove stopwords and make words lower.
    2. Combine numbers with particular symbols($, %).
        e.g. 1.98, % -> 1.98%    $, 543.27m -> $543.27m
    3. Remove words whose length equals to one.

    :param words: A list of words
    :type words: List
    :return: A list of normalized words
    :rtype: List

    '''
    nlz_words = [word.lower() for word in words
                 if word.lower() not in stopwords
                 ]
    
    nlz_words2 = []
    c = 0
    for i in range(len(nlz_words)):
        if c:
            c = 0
        elif i==len(nlz_words)-1:
            nlz_words2.append(nlz_words[i])
        elif nlz_words[i]=='$' and is_num(nlz_words[i+1]):
            nlz_words2.append(nlz_words[i]+nlz_words[i+1])
            c = 1
        elif nlz_words[i+1]=='%' and is_num(nlz_words[i]):
            nlz_words2.append(nlz_words[i]+nlz_words[i+1])
            c = 1
        else:
            nlz_words2.append(nlz_words[i])
          
    nlz_words3 = [word for word in nlz_words2
                  if not (len(word)==1 and not is_num(word))
                  ]
    
    return nlz_words3

text = data_collector.load_data('apple', 'keyword')
label = stock_data.stock_price_label('AAPL', 14, 5)
inputs = data_collector.merge_price_text(text, label).values
nlz_inputs = [([word for word in normalizing(words)], tuple(label))
              for (words, label) in inputs]
inputs = [(words, tuple(label)) for (words, label) in inputs]
all_words = list(itertools.chain(*[words for (words, _) in nlz_inputs]))
fd = FreqDist(all_words)
word_features = [word for (word, _) in fd.most_common(2000)]

def features_contain(words):
    '''
    A feature extractor whose features indicate whether or not individual
    words are present in a given words.

    return example:
    {'contain(apple)': True, 'contain(banana)': False, ...}

    :param words: A list of words
    :type words: List
    :return: Features that indicate whether or not individual words are present
    in a given words
    :rtype: Dict
    '''
    words = set(words)
    features = {}
    for word in word_features:
        features['contain({})'.format(word)] = (word in words)
    return features

def features_ratio(words):
    '''
    A feature extractor whose features indicate the ratio of individual words.

    return example:
    {'ratio(apple)': 0.2, 'ratio(banana)': 0.0, ...}

    :param words: A list of words
    :type words: List
    :return: Features that indicate the ratio of individual words
    :rtype: Dict
    '''
    features = {}
    for word in word_features:
        features['ratio({})'.format(word)] = 0.0
    for word in words:
        try:
            features['ratio({})'.format(word)] += 1/len(words)
        except KeyError:
            pass
    return features 

def combine_symbol(sent):
    result = ''
    c = 0
    for i in range(len(sent)):
        if c:
            c = 0
        elif i==len(sent)-1:
            result += sent[i]
        elif sent[i]=='$':
            result += sent[i]
            c = 1
        elif sent[i+1]=='%':
            pass
        else:
            result += sent[i]
    return result

def features_ratio_of_num_sents(words):
    sents = nltk.sent_tokenize(' '.join(words))
    sents = [combine_symbol(sent) for sent in sents]
    nlz_words = normalizing(words)
    num_words = {word for word in nlz_words if word.startswith('$') or
                 word.endswith('%')}
    num_sents = list({sent for sent in sents for num in num_words
                      if num in sent})
    new_words = nltk.word_tokenize(' '.join(num_sents))
    new_nlz_words = normalizing(new_words)
    return features_ratio(new_nlz_words)

num_sents = []
for (words, label) in inputs:
    sents = nltk.sent_tokenize(' '.join(words))
    sents = [combine_symbol(sent) for sent in sents]
    nlz_words = normalizing(words)
    num_words = {word for word in nlz_words if word.startswith('$') or
                 word.endswith('%')}
    num_sents.extend(list({sent for sent in sents for num in num_words
                           if num in sent}))
num_sents = list(set(num_sents))

words = []
for sent in num_sents:
    words.extend(nltk.word_tokenize(sent))
words = normalizing(words)
fd2 = FreqDist(words)

from collections import defaultdict

before = {word: fd[word]/fd.N() for word in fd}
after = defaultdict(float)
for word in before:
    after[word] = fd2[word]/fd2.N()

ratio = {word: after[word]/before[word] for word in before if not is_num(word)}
fd3 = FreqDist(ratio)

    

# 아래 주석 1, 2, 3 중 원하는 거 하나 지워주세요

#1 featuresets = text_processing.featureset
'''
text_processing.featureset example:
[({'mentioned vs not mentioned': 4.47, 'company mentioned polarity scores': 0.24,
'total polarity scores': 0.9968, 'word related to increase': 22,
'word related to decrease': 7, 'number of news': 0}, (1, 0, 0)), ...]
'''
'''
features = features_ratio
featuresets = [(features(words), label) for (words, label) in nlz_inputs]

#3 features = features_ratio_of_num_sents
#3 featuresets = [(features(words), label) for (words, label) in inputs]

random.shuffle(featuresets)
slicing_point = int(len(featuresets) * 0.1)

train_set, test_set = featuresets[slicing_point:], featuresets[:slicing_point]

start = time.time()
classifier = maxent.MaxentClassifier.train(train_set, 'gis', max_iter=10,
                                           trace=1)
'''                                        
'''
encoding = maxent.TypedMaxentFeatureEncoding.train(train_set,
                                                   alwayson_features=True)
classifier = maxent.MaxentClassifier.train(
    train_set, bernoulli=False, encoding=encoding, max_iter=10)
'''
'''
print('training time:', time.time()-start)
print()

test = [feature for (feature, _) in test_set]
labels = [label for (_, label) in test_set]
print((1, 0, 0), (0, 1, 0), (0, 0, 1), '  gold   correct?')
print('-----------------------------------------------')
cut = 0
for i in range(len(test)):
    if cut < 50:
        pdist = classifier.prob_classify(test[i])
        guess = classifier.classify(test[i])
        gold = labels[i]
        print('{:^9.4f} {:^9.4f} {:^9.4f}'.format(
            pdist.prob((1, 0, 0)), pdist.prob((0, 1, 0)), pdist.prob((0, 0, 1))
            ), end=' ')
        print(gold, guess==gold)
        cut += 1
    else:
        break
# print top 50 results

print('accuracy:', nltk.classify.accuracy(classifier, test_set))
'''
'''
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
x, y, z = [], [], []
for i in range(len(test)):
    pdist = classifier.prob_classify(test[i])
    x.append(pdist.prob((1, 0, 0)))
    y.append(pdist.prob((0, 1, 0)))
    z.append(pdist.prob((0, 0, 1)))
x, y, z = np.array(x), np.array(y), np.array(z)
ax = plt.axes(projection='3d')
ax.set_xlabel((1, 0, 0))
ax.set_ylabel((0, 1, 0))
ax.set_zlabel((0, 0, 1))
ax.set_xlim3d(0, 1)
ax.set_ylim3d(0, 1)
ax.set_zlim3d(0, 1)
ax.scatter(x, y, z)
plt.show()
'''
