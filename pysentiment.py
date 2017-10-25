import pandas as pd
import numpy as np
import yaml
import re
import os
import sys
from pprint import pprint
from nltk import *
import lda
import xml
import string
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
from scipy.sparse import coo_matrix
from gensim.models import word2vec
from nltk.stem import WordNetLemmatizer
from pprint import pprint
import gensim
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

pd.set_option('display.max_colwidth',-1)

class DictTagger:

    def __init__(self, xml_path, valence_path, score_path):
        
        self.score = {}
        self.valence = {}
        self.dictionary = {}
        xml_data = open(xml_path).read()
        root = ET.XML(xml_data)
        for i, child in enumerate(root):
            # print (i, child.text)
            text = child.text
            text = re.sub(r":", "", text)
            self.dictionary[i+1] = text
                
        # print(self.dictionary)
        topics = ['anger','disgust','fear','joy','sadness','surprise']
        with open(score_path, 'rb') as file:
            lines = file.readlines()
            for line in lines:
                tmp = [int(x) for x in line.split()]
                id = tmp[0]
                scores = tmp[1:]
                self.score[tmp[0]] = topics[scores.index(max(scores))]
            file.close()
        with open(valence_path, 'rb') as file:
            lines = file.readlines()
            for line in lines:
                tmp = [int(x) for x in line.split()]
                self.valence[tmp[0]] = tmp[1]
            file.close()


    def data_frame(self):

        df = pd.DataFrame(columns=['id','text','score','valence'])
        for key in sorted(self.dictionary.keys()):
            try:
                df = df.append({'id':key, 'text':self.dictionary[key], 'score':self.score[key], 'valence':self.valence[key]},ignore_index=True)
            except KeyError:
                pass
        return df

def remove_urls(text):
    text = re.sub(r"(?:\@|http?\://)\S+", "", text)
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    return text

def lemmanize(text):
    lemmatizer = WordNetLemmatizer()
    normalized = [lemmatizer.lemmatize(word) for word in word_tokenize(text)]
    return normalized

def stopwrds(sent):
    stopword = set(stopwords.words('english'))
    words = word_tokenize(sent)    
    stop_free = " ".join([w for w in words if w not in stopword])
    return stop_free

def exclude(stop_free):
    exclude = set(string.punctuation) 
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    return punc_free

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

if __name__ == '__main__':

    xml_path = "affectivetext_trial.xml"
    score_path = "affectivetext_trial.emotions.gold"
    valence_path = "affectivetext_trial.valence.gold"
    topics = ['anger','disgust','fear','joy','sadness','surprise']

    dictagger = DictTagger(xml_path, score_path, valence_path)
    df = dictagger.data_frame()
    df.to_csv("data.csv", index=False)

    # df['text'] = df['text'].apply(remove_urls)
    # df['text'] = df['text'].apply(stopwrds)
    # df['text'] = df['text'].apply(exclude)
    # df['text'] = df['text'].apply(lemmanize)
    # print (df.head())
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=1000,
                                   stop_words='english',
                                   ngram_range=(1,2))
    tfidf = tfidf_vectorizer.fit_transform(df.text)

    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=1000,
                                stop_words='english',
                                ngram_range=(1,2))
    tf = tf_vectorizer.fit_transform(df.text)


   
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    n_top_words = 10
    # print_top_words(nmf, tfidf_feature_names, n_top_words)

    lda = LatentDirichletAllocation(n_components=5, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
    lda.fit(tf)

    tf_feature_names = tf_vectorizer.get_feature_names()
    print(tfidf_feature_names)
    print_top_words(lda, tf_feature_names, n_top_words)


