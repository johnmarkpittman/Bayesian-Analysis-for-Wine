import numpy as np
import re
import nltk
from nltk.stem.snowball import SnowballStemmer 
from nltk.util import ngrams
from nltk.corpus import stopwords
stemmer = SnowballStemmer('english')

# =============================================================================
# Notes
# =============================================================================

# =============================================================================
# Text Cleaning
# =============================================================================
def remove_text(pattern, df, col): #uses regex to find and replace text in posts
    string = re.compile(r'{}'.format(pattern))
    removed_text = df[col].str.findall(string)
    df[col] = df[col].str.replace(string, '')
    return df

def drop_empty_posts(df, col):
    '''currently only accepts a single df and a list of columns to be dropped (based on col name).'''
    to_drop = np.where(df[col].isna() == True)[0]
    df = df.drop(labels = to_drop, axis = 0)
    to_drop = np.where(df[col] == '')[0]
    df = df.drop(labels = to_drop, axis = 0)
    return df

def remove_whitespace(df, col):
    # Lowercase
    df[col] = df[col].str.lower()
    # Remove new lines and tabs
    t = str.maketrans("\n\t\r", "   ")
    df[col] = df[col].str.translate(str.maketrans(t))
    df[col] = df[col].str.strip()
    # Remove Whitespace
    return df
#    return df

def text_to_lower(df, col):
    df[col] = df[col].str.lower()
    return df

def remove_punct(df, col):
    df[col] = df[col].str.replace('[^\w\s]','')
    return df

def remove_stopwords(df, col):
    stop = stopwords.words('english')
    df[col] = df[col].apply(lambda x: ' '.join(x for x in x.split() if x not in stop))
    return df

def tokenize_only(text):
    return [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    
def tokenize_and_stem(text):
    tokens = tokenize_only(text)
    return [stemmer.stem(t) for t in tokens]

def tokenize_and_stem_and_bigram(text):
    tokens = tokenize_and_stem(text)
    word_list = list(tokens)
    bigrams = list(ngrams(tokens, 2))
    bigram_list = []
    for bigram in bigrams:
        bigram_list.append(' '.join(bigram))
    word_list += bigram_list
    return word_list
    
def text_process_pipeline(df, col, quotes=True, links=True, PGP=True, punctuation=True, whitespace=True, lower=True, stopwords=True):
    if quotes is True:
        print('Removing "Quotes from: <author name> <date>" from column')
        df = remove_text('Quote.*?(?<=\d\d:\d\d\s)am|Quote.*?(?<=\d\d:\d\d\s)pm', df, col)
    if links is True:
        print('Removing web links from column')
        df = remove_text('(http.*?)\s', df, col)
    if PGP is True:
        print('Removing PGP keys, signatures, and messages from column')
        df = remove_text('-----BEGIN PGP PUBLIC KEY BLOCK-----[\s\S]*-----END PGP PUBLIC KEY BLOCK-----', df, col)
        df = remove_text('-----BEGIN PGP SIGNATURE-----[\s\S]*-----END PGP SIGNATURE-----', df, col)
        df = remove_text('-----BEGIN PGP MESSAGE-----[\s\S]*-----END PGP MESSAGE-----', df, col)
    if punctuation is True:
        print('Removing extra whitespace from column')
        df = remove_punct(df, col)
    if lower is True:
        print('Making all text in column lowercase')
        df = text_to_lower(df, col)
    if stopwords is True:
        print('Removing empty strings/nulls and stopwords from column')
        df = drop_empty_posts(df, col)
        df = remove_stopwords(df, col)
        df = drop_empty_posts(df, col)
    if whitespace is True:
        print('Removing extra whitespace from column')
        df = remove_punct(df, col)
    df.reset_index(inplace = True)
    df.rename(columns = {'index': 'post_id'}, inplace = True)
    return df
        
# =============================================================================
# Text Features
# =============================================================================

def avg_word_len(sentence):
    '''takes a single string as argument and returns avg word length'''
    try:
        words = sentence.split()
        return round((sum(len(word) for word in words)/len(words)), 2)
    except ZeroDivisionError:
        return 0 
    

