
#%%
#---------------------------------------------------PREPROCESSOR MODULE----------------------------------------------------------------
#
# Import packages
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import regex as re 
from copy import copy
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')

#%%
# Defining the stop words
stop = stopwords.words('english')
words_not_tobe_removed = ['it', 'its', 'they', 'them', 'themselves', 'their', 'what', 'which', 'this', 'that', 'these', 'those', 'be', 'being', 'have', 'has', 'having',
                         'do', 'does', 'did', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'there', 'by', 'for', 'with', 'about', 'against',
                          'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                          'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                          'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'don', "don't",
                          'should', "should've", 'now', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                          'haven', "haven't", 'isn', "isn't", 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
                          'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

final_stop_words = set([word for word in stop if word not in words_not_tobe_removed])


stemmer = SnowballStemmer("english")
# %%
def preProcess(dataframe, norm_method ='lemmatization'):
    """ input: dataframe --> input dataframe, with columns Title, Section and Content (out of the docParser)
        output: dataframe with additional column sentences_p_s, which contains the ready to biencode contents."""
    
    # copy the dataframe into temperory variable(to avoid modifying original dataframe)
    df = copy(dataframe)
    
    # change none and 'None' etc in any of the section columns as '' (fill all other empty cells also with the '')
    for col in df.columns:
        df[col] = df[col].apply(lambda x:'' if (x == 'None' or x == None) else x)
    df.fillna('', inplace=True)
    
    # remove additional dots, or underscore (TOC pages/form like pages)
    df['Content'] = df['Content'].apply(lambda x: re.sub('\.\.+ | \s\s+',' ',x))

    # add column length and remove the shorter sentences
    df['length'] = df['Content'].apply(lambda x: len(x))
    df = df[df['length'] > 7]

    # remove sentences with less than 4 words.
    df['word_num'] = df['Content'].apply(lambda x: len(x.split()))
    df = df[df['word_num'] >=  5] 

    
    # append section headers to the left of the sentences and make columns sentences_p_s
    df['sentences_p_s'] = df.Title + " " + df.Section + " " + df.Content
    
    # remove any special characters, non ascii letters etc.
    pattern='[^a-z^0-9^A-Z^[^\n^.^\-^,^:^&^$^(^)^/^|^%]'
    df['sentences_p_s'] = df['sentences_p_s'].apply(lambda x: re.sub(pattern,' ',x))
    
    # remove stop words
    df['sentences_p_s'] = df['sentences_p_s'].astype(str).apply(lambda x: ' '.join([word for word in x.split() if word not in (final_stop_words)]))
    
    # convert into small letters
    df['sentences_p_s'] = df['sentences_p_s'].str.lower()
    
    # strip additional spaces if any
    df['sentences_p_s'] = df['sentences_p_s'].str.strip()
    
    # apply stemming or lemmetisation
    if norm_method == 'stemming':
        df['sentences_p_s'] = df['sentences_p_s'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
    elif norm_method == 'lemmatization':
        df['sentences_p_s'] = df['sentences_p_s'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
        
    # modify the length column according to the encoded sentences
    df['length'] = df['sentences_p_s'].apply(lambda x: len(x))
        
    df.reset_index(inplace = True, drop=True)
    # return the new dataframe
    return df
# %%
