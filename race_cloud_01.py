# Created by R James on 5.14.20
# Last updated: N/A
# Description: Wordcloud from race data

### Importing packages

import pandas as pd
import re
import nltk
import numpy as np
from pprint import pprint

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import spacy

import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
%matplotlib inline

# Importing data

df = pd.read_csv('C:/Users/asus/Google Drive/Projects/Red Pill/Data/SOC560 Analysis/Results/RP_incels_topicsentences_race_01.csv')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['reddit', 'r', 'post', 'comment', 'sub','subreddit','user','http','hed','shed','dont','youre','thats','its','cant','wont','wouldnt','couldnt','shouldnt','havent', 'www', 'com'])

df.columns = ['index','Dominant_Topic','Perc_Contribution','Topic_Keywords','text']

data = df.text.values.tolist()

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

data_words_nostops = remove_stopwords(data_words)

long_string = ', '.join(map(str, data_words_nostops))

from wordcloud import WordCloud

wordcloud = WordCloud(background_color="white",max_words=100, width=2500, height=2000, contour_width=3,contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()
wordcloud.to_file('race_cloud_01.png')
