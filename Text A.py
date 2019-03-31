# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 15:58:06 2019

@author: nakul
"""

import numpy as np
import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import wordcloud
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')


nltk.download('punkt')
txtf = open("T1.txt","r")
txtf2 = open("T2.txt","r")
txtf3 = open("T3.txt","r")
txtf4 = open("T4.txt","r")
txtf5 = open("T5.txt","r")
txtf6 = open("T6.txt","r")
txtf7 = open("T7.txt","r")
txtf8 = open("T8.txt","r")
print(txtf.read(1235))
print(sent_tokeniz(txtf5))
for line in txtf4:
    print(line)
#Reading all files at once
    

adoc = txtf.read()
adoc2 = txtf2.read()
adoc3 = txtf3.read()
adoc4 = txtf4.read()
adoc5 = txtf5.read()
adoc6 = txtf6.read()
adoc7 = txtf7.read()
adoc8 = txtf8.read()
#COnvert all texts to lower case letters
adb = ("%s" %adoc).lower()
adb2 = ("%s" %adoc2).lower()
adb3 = ("%s" %adoc3).lower()
adb4 = ("%s" %adoc4).lower()
adb5 = ("%s" %adoc5).lower()
adb6 = ("%s" %adoc6).lower()
adb7 = ("%s" %adoc7).lower()
adb8 = ("%s" %adoc8).lower()

so = adc+adc2
len(so)
adc = adb.replace("n't", "not")
adc= adc.replace('-', ' ')
adc = adc.replace('_', ' ')
adc = adc.replace(',', ' ')

adc2 = adb2.replace("n't", "not")
adc2= adc2.replace('-', ' ')
adc2 = adc2.replace('_', ' ')
adc2 = adc2.replace(',', ' ')


adc3 = adb3.replace("n't", "not")
adc3= adc3.replace('-', ' ')
adc3 = adc3.replace('_', ' ')
adc3 = adc3.replace(',', ' ')

adc4 = adb4.replace("n't", "not")
adc4= adc4.replace('-', ' ')
adc4 = adc4.replace('_', ' ')
adc4 = adc4.replace(',', ' ')


adc5 = adb5.replace("n't", "not")
adc5= adc5.replace('-', ' ')
adc5 = adc5.replace('_', ' ')
adc5 = adc5.replace(',', ' ')

adc6 = adb6.replace("n't", "not")
adc6= adc6.replace('-', ' ')
adc6 = adc6.replace('_', ' ')
adc6 = adc6.replace(',', ' ')


adc7 = adb7.replace("n't", "not")
adc7= adc7.replace('-', ' ')
adc7 = adc7.replace('_', ' ')
adc7 = adc7.replace(',', ' ')

adc8 = adb8.replace("n't", "not")
adc8= adc8.replace('-', ' ')
adc8 = adc8.replace('_', ' ')
adc8 = adc8.replace(',', ' ')

tokens = word_tokenize(adc)
tokens2 = word_tokenize(adc2)
tokens3 = word_tokenize(adc3)
tokens4 = word_tokenize(adc4)
tokens5 = word_tokenize(adc5)
tokens6 = word_tokenize(adc6)
tokens7 = word_tokenize(adc7)
tokens8 = word_tokenize(adc8)
tzd = tokens+tokens2+tokens3+tokens4+tokens5+tokens6+tokens7+tokens8
len(tzd)
token = pos_tag(tokens)
token2 = pos_tag(tokens2)
token3 = pos_tag(tokens3)
token4 = pos_tag(tokens4)
token5 = pos_tag(tokens5)
token6 = pos_tag(tokens6)
token7 = pos_tag(tokens7)
token8 = pos_tag(tokens8)

print(len(token))
print(len(token2))
print(len(token3))
print(len(token4))
print(len(token5))
print(len(token6))
print(len(token7))
print(len(token8))

tok = token+token2+token3+token4+token5+token6+token7+token8
len(tok)



#POS
tagged_tokens = nltk.pos_tag(tzd)
pos_list = [word[1] for word in tagged_tokens if word[1] != ":" and word[1] != "."]
from nltk.probability import FreqDist
pos_dist = FreqDist(pos_list)
pos_dist.plot(title="Parts of Speech")
for pos, frequency in pos_dist.most_common(pos_dist.N()):
    print('{:<15s}:{:>4d}'.format(pos, frequency))
    
#Scenario 1: Stopwords Yes, 
stop = stopwords.words('english') + list(string.punctuation)
stop_tokens = [word for word in tagged_tokens if word[0] not in stop]
# Remove single character words and simple punctuation
stop_tokens = [word for word in stop_tokens if len(word) > 1]
# Remove numbers and possive "'s"
stop_tokens = [word for word in stop_tokens \
if (not word[0].replace('.','',1).isnumeric()) and \
word[0]!="'s" ]
token_dist = FreqDist(stop_tokens)
print("\nCorpus contains", len(token_dist.items()), \
" unique terms after removing stop words.\n")
for word, frequency in token_dist.most_common(20):
    print('{:<15s}:{:>4d}'.format(word[0], frequency))
    
#STemming
stemmer = SnowballStemmer("english")
wn_tags = {'N':wn.NOUN, 'J':wn.ADJ, 'V':wn.VERB, 'R':wn.ADV}
wnl = WordNetLemmatizer()
stemmed_tokens = []
for token in stop_tokens:
    term = token[0]
    pos = token[1]
    pos = pos[0]
    try:
        pos = wn_tags[pos]
        stemmed_tokens.append(wnl.lemmatize(term, pos=pos))
    except:
        stemmed_tokens.append(stemmer.stem(term))
# Get token distribution
fdist = FreqDist(stemmed_tokens)
print("\nCorpus contains", len(fdist.items()), \
" unique terms after Stemming.\n")

for word, freq in fdist.most_common(20):
    print('{:<15s}:{:>4d}'.format(word, freq))
fdist_top = nltk.probability.FreqDist()
for word, freq in fdist.most_common(20):
    fdist_top[word] = freq
fdist_top.plot()

#Scennario 2: 
stop = stopwords.words('english') + list(string.punctuation)
stop_tokens = [word for word in tzd if word[0] not in stop]
# Remove single character words and simple punctuation
stop_tokens = [word for word in stop_tokens if len(word) > 1]
# Remove numbers and possive "'s"
stop_tokens = [word for word in stop_tokens \
if (not word[0].replace('.','',1).isnumeric()) and \
word[0]!="'s" ]
token_dist = FreqDist(stop_tokens)
print("\nCorpus contains", len(token_dist.items()), \
" unique terms after removing stop words.\n")
for word, frequency in token_dist.most_common(20):
    print('{:<15s}:{:>4d}'.format(word[0], frequency))
    
stemmer = SnowballStemmer("english")
wn_tags = {'N':wn.NOUN, 'J':wn.ADJ, 'V':wn.VERB, 'R':wn.ADV}
wnl = WordNetLemmatizer()
stemmed_tokens = []
for token in stop_tokens:
    term = token[0]
    pos = token[1]
    pos = pos[0]
    try:
        pos = wn_tags[pos]
        stemmed_tokens.append(wnl.lemmatize(term, pos=pos))
    except:
        stemmed_tokens.append(stemmer.stem(term))
# Get token distribution
fdist = FreqDist(stemmed_tokens)
print("\nCorpus contains", len(fdist.items()), \
" unique terms after Stemming.\n")

for word, freq in fdist.most_common(20):
    print('{:<15s}:{:>4d}'.format(word, freq))
fdist_top = nltk.probability.FreqDist()
for word, freq in fdist.most_common(20):
    fdist_top[word] = freq
fdist_top.plot()

#Scenario 3
stop = stopwords.words('english') + list(string.punctuation)
stop_tokens = [word for word in tzd if word[0] not in stop]
# Remove single character words and simple punctuation
stop_tokens = [word for word in stop_tokens if len(word) > 1]
# Remove numbers and possive "'s"
stop_tokens = [word for word in stop_tokens \
if (not word[0].replace('.','',1).isnumeric()) and \
word[0]!="'s" ]
token_dist = FreqDist(stop_tokens)
print("\nCorpus contains", len(token_dist.items()), \
" unique terms after removing stop words.\n")
for word, frequency in token_dist.most_common(20):
    print('{:<15s}:{:>4d}'.format(word[0], frequency))
    

fdist = FreqDist(stop_tokens)
for word, freq in fdist.most_common(20):
    print('{:<15s}:{:>4d}'.format(word, freq))
fdist_top = nltk.probability.FreqDist()
for word, freq in fdist.most_common(20):
    fdist_top[word] = freq
fdist_top.plot()

#Scenario 4
fdist = FreqDist(tzd)
print("\nCorpus contains", len(fdist.items()), \
" unique terms without Stemming or removing Stopwords.\n")
for word, freq in fdist.most_common(20):
    print('{:<15s}:{:>4d}'.format(word, freq))
fdist_top = nltk.probability.FreqDist()
for word, freq in fdist.most_common(20):
    fdist_top[word] = freq
fdist_top.plot()


