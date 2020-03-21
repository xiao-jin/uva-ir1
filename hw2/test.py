import read_ap
from gensim.test.utils import common_dictionary, common_corpus
from gensim.models import LsiModel

model = LsiModel(common_corpus, id2word=common_dictionary)
vectorized_corpus = model[common_corpus]

docs = read_ap.get_processed_docs()

print(len(docs))

pass

"""
tf-idf > w2v > lsa  >> d2v
"""