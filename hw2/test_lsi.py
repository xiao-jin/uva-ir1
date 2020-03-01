import sys
import time
import logging
import doc_processor
import read_ap
import download_ap
from tqdm import tqdm
from gensim.corpora import Dictionary
from gensim import similarities
from gensim.models import LsiModel
import pytrec_eval

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

doc_list = doc_processor.get_doc_list()
print('Number of docs:', len(doc_list))

counter = 0
for doc in doc_list:
    counter += len(doc)
print('Total tokens:', counter)

freq_tokens = doc_processor.get_frequent_tokens(doc_list)
docs_matrix = doc_processor.remove_freqent_tokens(doc_list, freq_tokens)
print('Number of documents with infrequently tokens:', len(docs_matrix))

counter = 0
for v in docs_matrix:
    counter += len(v)
print('Number of infrequent tokens:', counter)

# Create gensim dictionaries
print('Create dictionary')
dictionary = Dictionary(docs_matrix)

# corpus = doc_processor.create_corpus(dictionary, doc_list, is_tfidf=False)
# tfidf_corpus = doc_processor.create_corpus(dictionary, doc_list, is_tfidf=True)

# ensure dataset is downloaded
download_ap.download_dataset()
# pre-process the text
docs_by_id = read_ap.get_processed_docs()

# Create instance for retrieval

# read in the qrels
qrels, queries = read_ap.read_qrels()

# print(lsi_model.show_topics())

overall_ser = {}

def test(mode):
    lsi_model = LsiModel.load('./test_models/lsi_model_{}.pt'.format(mode))
    print("Running %s Benchmark" % mode.upper())
    # collect results
    for qid in tqdm(qrels): 
        query_text = queries[qid]
        vec_qry = dictionary.doc2bow(query_text.lower().split())
        # print(query_text)
        results = lsi_model[vec_qry]
        # print(results)
        overall_ser[qid] = dict(results)

    # run evaluation with `qrels` as the ground truth relevance judgements
    # here, we are measuring MAP and NDCG, but this can be changed to 
    # whatever you prefer
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
    metrics = evaluator.evaluate(overall_ser)

    # dump this to JSON
    # *Not* Optional - This is submitted in the assignment!
    with open("lsi-{}.json".format(mode), "w") as writer:
        json.dump(metrics, writer, indent=1)

modes = ['bow', 'tfidf']

for mode in modes:
    test(mode)

# lsi_model_tfidf = LsiModel.load('./test_models/lsi_model_tfidf.pt')