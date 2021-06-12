#1.Tokeniation
#2.Stemming
#3.Lower case
#4.into vector

import nltk
nltk.download('punkt')
import numpy as np
from nltk.stem.porter import PorterStemmer
import gensim
from gensim.models import Word2Vec

stemmer = PorterStemmer()

def tokenize(sent):
    return nltk.word_tokenize(sent)

def stem(word):
    """It takes one word and returns the stem of the word."""
    return stemmer.stem(word.lower())

def vectorize(sent,total_words):
    vec = np.zeros(len(total_words))
    sent = [stem(w) for w in sent]
    for i,w in enumerate(total_words):
        if w in sent:
            vec[i] = 1.0

    return vec


