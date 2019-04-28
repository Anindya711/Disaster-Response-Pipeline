#importing packages
import numpy as np
import pandas as pd
import nltk
nltk.download(['wordnet', 'punkt', 'stopwords'])
#importing stopwords to get a list of stopwords
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

from sklearn.base import BaseEstimator, TransformerMixin
#this class will calculate the average word length in a text and return a
#dataframe with all the average word lengths in that series
class AverageWordLengthExtractor(BaseEstimator, TransformerMixin):

    '''
        Custom transformer class that have a transform method which returns
        the average world length in a pandas series of texts as a DataFrame
        which can be used a in a pipeline.
    '''

    def __init__(self):
        pass
    def average_word_length(self, text):
        '''
        calculates the average word length in a text.
        input : text (str)
        output : length (float)
        '''

        return np.mean([len(word) for word in text.split( ) if word not in stopWords])

    def fit(self, x, y=None):
        return self

    def transform(self, x , y=None):
        '''
            Takes as input a series of text from a input and returns the average
            word length in that series.
            input : pandas series of texts
            output : DataFrame with the average length of words for each text
        '''
        return pd.DataFrame(pd.Series(x).apply(self.average_word_length)).fillna(0)
