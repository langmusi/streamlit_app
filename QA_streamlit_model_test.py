# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 15:27:12 2021

@author: minli
"""

import streamlit as st

import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder, util

from rank_bm25 import BM25Okapi
#from sklearn.feature_extraction import stop_words
import string
from tqdm.autonotebook import tqdm
import numpy as np

st.title("Information Searching in A Given Context Demo")

# "C:/Users/LIUM3478/OneDrive Corp/OneDrive - Atkins Ltd/Work_Atkins/textanalys/data/close_defects_small.csv"
embeddings_filepath = 'D:/python_working_dir/nlp/data/close_defects_small.csv'

# caching data. Only run once if the data has not been loaded
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv(embeddings_filepath, encoding = 'utf-8', sep = ";")
    return df

# Will only run once if already cached
df = load_data()

# filling nan
df[['Skadebeskrivning', 'Skaderubrik', 'Åtgärdsbeskrivning']] = df[['Skadebeskrivning','Skaderubrik', 'Åtgärdsbeskrivning']].fillna(value='')
df['Skade_text'] = df['Skaderubrik'] + ' ' + df['Skadebeskrivning']
# droping nan
df = df[df['Skade_text'].notna()]
# removing multiple blanks
df['Skade_text'] = df['Skade_text'].replace('\s+', ' ', regex=True)
# removing trailing space of column in pandas
df['Skade_text'] = df['Skade_text'].str.rstrip()
passages = df['Skade_text'].values.tolist()


# We lower case our text and remove stop-words from indexing
def bm25_tokenizer(text):
  tokenized_doc = []
  for token in text.lower().split():
    token = token.strip(string.punctuation)

    #if len(token) > 0 and token not in stop_words.ENGLISH_STOP_WORDS:
    tokenized_doc.append(token)
  return tokenized_doc

tokenized_corpus = []

for passage in tqdm(passages):
  tokenized_corpus.append(bm25_tokenizer(passage))

bm25 = BM25Okapi(tokenized_corpus)

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 07:46:09 2021

@author: minli
"""

#This function will use lexical search all texts in passages that answer the query
def model_lexical_search(query, top_n_res):
    
    #BM25 search (lexical search)
    tokenized_corpus = []
    for passage in tqdm(passages):
      tokenized_corpus.append(bm25_tokenizer(passage))

    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n_res = int(top_n_res)
    top_n = np.argpartition(bm25_scores, -top_n_res)[-top_n_res:]
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)

    
    bm25_output = []

    print("Top-" + str(top_n_res) + "lexical search (BM25) hits")
    for hit in bm25_hits[0:top_n_res]:
        line = str(round(hit['score'], 2)) + " , " + passages[hit['corpus_id']]
        bm25_output.append(line)


    return bm25_output

###############
#This function will search all texts in passages that answer the query: ##### Sematic Search #####
def model_semantic_search(query, bi_encoder_name, top_k_biencoder, top_n_res):
    
    bi_encoder = SentenceTransformer(bi_encoder_name)
    top_k = int(top_k_biencoder)     #Number of passages we want to retrieve with the bi-encoder

    #Here, we compute the corpus_embeddings from scratch (which can take a while depending on the GPU)
    corpus_embeddings = bi_encoder.encode(passages,  batch_size=32, convert_to_tensor=True, show_progress_bar=True)
    
    
    ##### Sematic Search #####
    #Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    question_embedding = question_embedding.cuda()
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    
    #Output of top-10 hits
    print("Top-" + str(top_n_res) + "Bi-Encoder Retrieval hits")
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    bi_encoder_output = []
    for hit in hits[0:top_n_res]:
        line_bi = str(round(hit['score'], 2)) + " , " + passages[hit['corpus_id']] #+ " . " + hit['corpus_id']
        bi_encoder_output.append(line_bi)

    
    return bi_encoder_output


##############################
#This function will search all texts in passages that answer the query
def model_semantic_search_rerank(query, bi_encoder_name, cross_encoder_name, top_k_biencoder, top_n_res):
    
    bi_encoder = SentenceTransformer(bi_encoder_name)
    top_k = int(top_k_biencoder)     #Number of passages we want to retrieve with the bi-encoder

    #The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
    cross_encoder = CrossEncoder(cross_encoder_name)
    
    #Here, we compute the corpus_embeddings from scratch (which can take a while depending on the GPU)
    corpus_embeddings = bi_encoder.encode(passages,  batch_size=32, convert_to_tensor=True, show_progress_bar=True)
    
    
    ##### Sematic Search #####
    #Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    question_embedding = question_embedding.cuda()
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    ##### Re-Ranking #####
    #Now, score all retrieved passages with the cross_encoder
    cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    #Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]


    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    
    
    print("Top-" + str(top_n_res) + "Cross-Encoder Re-ranker hits")
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    cross_encoder_output = []
    for hit in hits[0:top_n_res]:
        line_c = str(round(hit['cross-score'], 2)) + passages[hit['corpus_id']] + '  ' + "  Åtgärdskod: " + df.Åtgärdskod[hit['corpus_id']] + "  Åtgärder: " + df.Åtgärdsbeskrivning[hit['corpus_id']] + ' '
        cross_encoder_output.append(line_c)
               
    return cross_encoder_output


bi_encoder_option = st.sidebar.selectbox(
    'Bi Encoder Mdoel list',
    ('paraphrase-multilingual-mpnet-base-v2', 'sentence-transformers/stsb-xlm-r-multilingual', 
     'sentence-transformers/msmarco-distilbert-multilingual-en-de-v2-tmp-lng-aligned'))

#st.write('You selected:', bi_encoder_option)


cross_encoder_option = st.sidebar.selectbox(
    'Cross Encoder Mdoel list',
    ("cross-encoder/ms-marco-TinyBERT-L-6", 
     "cross-encoder/quora-roberta-large", 
     "cross-encoder/qnli-electra-base",
     "cross-encoder/stsb-roberta-large"))


#st.write('You selected:', cross_encoder_option)

top_k = 100     #Number of passages we want to retrieve with the bi-encoder
top_number_output = st.sidebar.slider('Choose a value to show top number of outputs', 0, 10)

query = st.text_input("Please write your question/text here")

search_type_option = ("Lexical Search", "Semantic Search", "Semantic Search and Re-ranking")
search_type = st.sidebar.selectbox("Searching Options:" , search_type_option)


if search_type == "Lexical Search":
    st.write(model_lexical_search(query, top_n_res=top_number_output))
elif search_type == "Semantic Search":
    st.markdown(model_semantic_search(query, 
                             bi_encoder_name = bi_encoder_option, 
                             top_k_biencoder=top_k, top_n_res=top_number_output))
elif search_type == "Semantic Search and Re-ranking":
    st.write(model_semantic_search_rerank(query, 
                      bi_encoder_name = bi_encoder_option, 
                      cross_encoder_name = cross_encoder_option,  
                      top_k_biencoder=top_k, top_n_res=top_number_output))


