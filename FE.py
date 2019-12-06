
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.feature_extraction import text
from sklearn.preprocessing import OneHotEncoder
from nltk.corpus import stopwords
import nltk.stem
import json

stemmer = nltk.stem.SnowballStemmer('english')

# hacking sklearn's CountVectorizer class to include stemming support

class StemmedCountVectorizer(text.CountVectorizer):

    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

def FE(wine, country_cutoff=1000, one_hot=False):

    # Opening wine and country dictionaries
    with open('country_dict.json', 'r') as fp:
        country_dict = json.load(fp)

    with open('wine_dict.json', 'r') as fp:
        wine_dict = json.load(fp)

    wine.set_index('id', inplace=True)

    # creating 'year' feature
    wine['year'] = wine['title'].str.extract(r'([1-9][0-9][0-9][0-9])')

    # creating 'continent' feature
    wine['continent'] = wine['country'].map(country_dict)
    
    # condensing 'country' column values
    countries = wine[['country', 'title']].groupby('country').count()
    popular_countries = countries[countries['title'] > country_cutoff].index.tolist()
    wine['country'] = np.where(wine['country'].str.contains('|'.join(popular_countries)), wine['country'], 'Other')

    # creating 'color' feature
    wine['color'] = wine['variety'].map(wine_dict)

    # creating price 'category' feature
    wine['category'] = np.where(wine['price'] <= 15, 0,
                                np.where(wine['price'] <= 30, 1,
                                         np.where(wine['price'] <= 50, 2, 3)))

    # creating score categorical feature
    wine['score_descriptive'] = np.where(wine['points'] <= 80, 'bad',
                                    np.where(wine['points'] <= 82, 'Acceptable',
                                         np.where(wine['points'] <= 86, 'Good', 
                                                 np.where(wine['points'] <= 89, 'Very Good',
                                                         np.where(wine['points'] <= 93, 'Excellent',
                                                                 np.where(wine['points'] <= 97, 'Superb','Classic',))))))

    # Creating TF-Matrix
    my_stop_words = text.ENGLISH_STOP_WORDS.union(['country', 'color'])
    wine = wine.copy(deep=True).reset_index()
    wine['description'] = wine['description'].str.replace('\d+', '')
    vectorizer = StemmedCountVectorizer(analyzer="word", stop_words=my_stop_words)
    processed_reviews = pd.DataFrame.sparse.from_spmatrix(vectorizer.fit_transform(wine['description']))
    new_cols = list(wine.columns) + list(vectorizer.vocabulary_.keys())
    
    # Converting Variety to One_hot_encoding
    if one_hot is True:
        encoder = OneHotEncoder()
        encoded = pd.DataFrame.sparse.from_spmatrix(encoder.fit_transform(wine[['color']]))
        wine.drop(columns=['color'], inplace=True)
        new_cols = list(wine.columns) + list(vectorizer.vocabulary_.keys())

        if (wine.shape[0] == processed_reviews.shape[0]) and (wine.shape[0] == encoded.shape[0]):
            wine_merged = pd.concat([wine, processed_reviews, encoded], axis=1, ignore_index=True)
            wine_merged.columns = new_cols + [k.replace('-', '_').replace(' ', '_').lower() for k in encoder.get_feature_names()]
        else:
            wine_merged = 'mismatched indices, could not properly concatenate'

    else:
        if wine.shape[0] == processed_reviews.shape[0]:
            wine_merged = pd.concat([wine, processed_reviews], axis=1, ignore_index=True)
            wine_merged.columns = new_cols
    
        else:
            wine_merged = 'mismatched indices, could not properly concatenate'

    return wine_merged

