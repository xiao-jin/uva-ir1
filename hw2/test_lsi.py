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
id2word = dictionary.id2token

corpus = doc_processor.create_corpus(dictionary, doc_list, is_tfidf=False)
tfidf_corpus = doc_processor.create_corpus(dictionary, doc_list, is_tfidf=True)


lsi_model = LsiModel.load('./saved_models/lsi_model_bow')
# lsi_model_tfidf = LsiModel.load('./saved_models/lsi_model_tfidf')
# lsi_model.id2word = id2word


# index = similarities.MatrixSimilarity(lsi_model[corpus]) 

# ensure dataset is downloaded
download_ap.download_dataset()
# pre-process the text
docs_by_id = read_ap.get_processed_docs()

# Create instance for retrieval

# read in the qrels
qrels, queries = read_ap.read_qrels()

print(lsi_model.show_topics())

overall_ser = {}

print("Running TFIDF Benchmark")
# collect results
for qid in tqdm(qrels): 
    query_text = queries[qid]
    vec_qry = dictionary.doc2bow(query_text.lower().split())
    print(query_text)
    results = lsi_model[vec_qry]
    overall_ser[qid] = dict(results)

# run evaluation with `qrels` as the ground truth relevance judgements
# here, we are measuring MAP and NDCG, but this can be changed to 
# whatever you prefer
# evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
metrics = evaluator.evaluate(overall_ser)

# dump this to JSON
# *Not* Optional - This is submitted in the assignment!
with open("tf-idf.json", "w") as writer:
    json.dump(metrics, writer, indent=1)

pass
