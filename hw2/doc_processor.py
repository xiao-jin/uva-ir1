# import packages
import read_ap
from tqdm import tqdm
from gensim.test.utils import common_dictionary, common_corpus
from gensim.corpora import Dictionary
from gensim.models import LsiModel
from gensim.models import LdaModel
from gensim.sklearn_api import TfIdfTransformer

def get_doc_list():
    """
    Process documents and convert doc Dictionary to a list of lists of tokens
    """
    docs = read_ap.get_processed_docs()
    return list(map(list, docs.values()))


def get_frequent_tokens(docs, min_threshold=50):
    tokens = {}
    
    for doc in tqdm(docs):
        for token in doc:
            if token in tokens:
                tokens[token] += 1
            else:
                tokens[token] = 1
                
    return [key for (key, value) in tokens.items() if value > min_threshold]


def remove_freqent_tokens(docs, freq_tokens):
    """
    Remove the most frequence tokens from the docs
    Param: docs, a list of lists of tokens
    Param: freq_token, a list of tokens
    Returns a list of lists of the tokens
    
    """
    docs_matrix = []

    tokenset = set(freq_tokens)
    for doc in tqdm(docs):
        infreq_tokens = list(set(doc) - tokenset)
        if len(infreq_tokens) > 0:
            docs_matrix.append(infreq_tokens)
    
    return docs_matrix

    
def create_corpus(dictionary, docs, is_tfidf=False):
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    
    if is_tfidf:
        corpus = convert_corpus(dictionary, corpus)

    return corpus


def convert_corpus(dictionary, corpus):
    # Transform the word counts inversely to their global frequency using the sklearn interface.
    model = TfIdfTransformer(dictionary=dictionary)
    tfidf_corpus = model.fit_transform(corpus)
    return tfidf_corpus