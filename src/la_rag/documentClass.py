import pandas as pd
import numpy as np

# import nltk
from copy import copy
# import transformers
import torch
import numpy as np
from sentence_transformers import CrossEncoder, util
from sentence_transformers import SentenceTransformer
import sys
# sys.path.append(r'D:\Muafira\Desktop\Complete RFP Analyser Pipeline\AI-DOC-PARSER') # path to the AI-DOC-PARSER
# sys.path.append(r'D:\Muafira\Desktop\Complete RFP Analyser Pipeline\RFP-PREPROCESSOR') # path to the RFP-PREPROCESSOR
from .layout_parser import docParser
from .preprocessor import preProcess
import torch
# from transformers import AutoTokenizer, AutoModel
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')


def cleanDf(df):
    try:
        df.drop(columns='Unnamed: 0', inplace=True)
    except:
        pass
    df.fillna('',inplace=True)
    df.drop_duplicates(subset = ['Content','Title','Section'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def biencodeDf(df):
    dataset = copy(df)
    bi_embeddings2 = bi_encoder.encode(dataset['sentences_p_s'], convert_to_tensor=True)
    dataset['biencoder'] = np.array(bi_embeddings2.cpu()).tolist()
    biembeddings = torch.tensor(bi_embeddings2)
    return  dataset, biembeddings

def search_data(dataframe,biembeddings, userQuery):
    
    if dataframe is not None and userQuery is not None:
        df = dataframe.copy()
        query = bi_encoder.encode([userQuery], convert_to_tensor=True)
        hits = util.semantic_search(query, biembeddings, top_k = 30)
        hits = hits[0]
        cross_inp = [[userQuery, dataframe.sentences_p_s.values[hit['corpus_id']]] for hit in hits]
        cross_scores = cross_encoder.predict(cross_inp)
        for idx in range(len(cross_scores)):
            hits[idx]['cross-score'] = cross_scores[idx]
        hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
        matches = []
        if hits[0]['cross-score'] < -5:
            matches.append({'Result':'No Match Found'})
            return pd.DataFrame(matches)
        for h_count,hit in enumerate(hits):
            if hit['cross-score'] >= -5:
                matches.append({'Section' : df.Section[hit['corpus_id']],
                            'Content' : df.Content.values[hit['corpus_id']],
                            'Page': df.Page.values[hit['corpus_id']]+1,
                                'Search_score': hit['cross-score']
                           })
            else:
                return pd.DataFrame(matches)
    return pd.DataFrame(matches)

# class for searchDocument, 
class SearchDocument:
    # Initialise the model
    def __init__(self,document):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name)
        self.df = None
        self.pre_processed = None
        self.document = document
        self.embeddings = None
        self.bi_encoding = None
        
    # pre process the model
    def preprocess(self):
        # Get the structured dataframe out of document
        self.df = docParser(self.document)
        # preprocess the dataframe to make it suitbale for biencoding
        self.pre_processed = preProcess(self.df)
        self.pre_processed = cleanDf(self.pre_processed)
        # Apply biencoding
        self.bi_encoding,self.embeddings = biencodeDf(self.pre_processed)
        return self.bi_encoding,self.embeddings
    
    # Search results
    def searchResult(self,search_query):
#         print(len(self.bi_encoding))
#         print(len(self.embeddings))
        return search_data(self.bi_encoding,self.embeddings,search_query)
     