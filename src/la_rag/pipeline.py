# A. Import the required packages

import pandas as pd
import sys
import concurrent.futures
from copy import copy
import torch
import numpy as np
from sentence_transformers import CrossEncoder, util
from sentence_transformers import SentenceTransformer
import sys
from .layout_parser import docParser
from .preprocessor import preProcess
# Encoders
import torch
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
# semantic search model and embedding
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
model[1].word_embedding_dimension = 786

#---------------------------------------------------------
# B. General Functions Definitions

# B.1 Clean the text extraction result df
def cleanDf(df):
    try:
        df.drop(columns='Unnamed: 0', inplace=True)
    except:
        pass
    df.fillna('',inplace=True)
    df.drop_duplicates(subset = ['Content','Document','Title','Section'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

# B.2 Biencoding 
def biencodeDf2(df):
    dataset = copy(df)
    bi_embeddings2 = bi_encoder.encode(dataset['sentences_p_s'], convert_to_tensor=True)
    dataset['biencoder'] = np.array(bi_embeddings2.cpu()).tolist()
    biembeddings = torch.tensor(bi_embeddings2)
    return  dataset, biembeddings


# B.3 Retriever using semantic search
def retriever(dataframe,biembeddings, userQuery):
    
    if dataframe is not None and userQuery is not None:
        df = dataframe.copy()
        query = bi_encoder.encode([userQuery], convert_to_tensor=True)
        hits = util.semantic_search(query, biembeddings, top_k = 50)
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
                            'Document' : df.Document.values[hit['corpus_id']],
                            'Page': df.Page.values[hit['corpus_id']]+1,
                                'Search_score': hit['cross-score']
                           })
            else:
                return pd.DataFrame(matches)
    return pd.DataFrame(matches)


# B.4 call openai
def findAnswer(context, query):
    # prompt_template = f"You are an extractive qa model gives answer to given query. You are given a query and a set of context.You have to provide specific answer from the given context, give your answer based only on the context. DON'T generate an answer that is NOT given in the provided context.If you don't find the answer within the context provided say 'No answers found!' .Use bullet points if you have to make a list, only if necessary. Give source of truth for each extracted information in bracket as number of item in context.QUERY: Give the {query}CONTEXTS:========={context}========="
    # response = openai.Completion.create(
    #         prompt=prompt_template,
    #         temperature=0,
    #         max_tokens=300,
    #         top_p=1,
    #         frequency_penalty=0,
    #         presence_penalty=0,
    #         model=COMPLETIONS_MODEL
    #     )
    # answer = response['choices'][0]['text']
    answer = "No Answer Found! /n Reason: Technical Issue from OpenAI model"
    return answer

def callOpenAI(context, query):
    # prompt_template = f"You are an extractive qa model gives answer to given query. You are given a query and a set of context.You have to provide specific answer from the given context, give your answer based only on the context. DON'T generate an answer that is NOT given in the provided context.If you don't find the answer within the context provided say 'No answers found!' .Use bullet points if you have to make a list, only if necessary. Give source of truth for each extracted information in bracket as number of item in context.QUERY: Give the {query}CONTEXTS:========={context}========="
    # response = openai.Completion.create(
    #         prompt=prompt_template,
    #         temperature=0,
    #         max_tokens=300,
    #         top_p=1,
    #         frequency_penalty=0,
    #         presence_penalty=0,
    #         model=COMPLETIONS_MODEL
    #     )
    # answer = response['choices'][0]['text']
    # answer = "No Answer Found! /n Reason: Technical Issue from OpenAI model"
    return answer
#_________________________________________________________________________
class SearchDocuments:
    def __init__(self, doc_paths, doc_names,search_query):
        self.df = None
        self.pre_processed = None
        self.documents = doc_paths
        self.names = doc_names
        self.embeddings = None
        self.bi_encoding = None
        self.search_query = search_query
    def extract(self):
        dfs = []
        for index,doc in enumerate(self.documents):
            temp = docParser(doc)
            temp['Document'] = [self.names[index] for i in range(len(temp))]
            dfs.append(temp)
        self.df = pd.concat(dfs)
        return self.df
    def preprocess(self):
        self.pre_processed = preProcess(self.df)
        self.pre_processed = cleanDf(self.pre_processed)
        # Apply biencoding
        self.bi_encoding,self.embeddings = biencodeDf2(self.pre_processed)
        return self.bi_encoding,self.embeddings
     # Search results
    def searchResult(self):
#         print(len(self.bi_encoding))
#         print(len(self.embeddings))
        return retriever(self.bi_encoding,self.embeddings,self.search_query)
    
# C.2  Extract the Inofrmation 
def extractInfo(doc_paths, doc_names,query):
    # initialise the search model with the document
    searchmodel = SearchDocuments(doc_paths,doc_names,query)
    # preprocess the documents
    extracted = searchmodel. extract()
    preprocessed = searchmodel.preprocess()
    # retrieval for the given query
    context = searchmodel.searchResult()

    return preprocessed, extracted, context
