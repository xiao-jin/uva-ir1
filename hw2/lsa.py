# """
# use gensim to get a vocabulary from the processed docs => Dictionary

# pass to gensim's bow or tfidf functions => corpus

# pass corpus to gensim models
# """

import sys
import time
import logging
import doc_processor
import read_ap
from gensim.corpora import Dictionary
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


def train_lsa(is_tfidf, num_topics):
    # Create corpus
    print('Create corpus')
    corpus = doc_processor.create_corpus(dictionary, doc_list, is_tfidf)

    # Set training parameters.
    num_topics = num_topics
    chunksize = 20000

    start = time.time()
    id2word = dictionary.id2token
    print('Start LSI training')

    lsi_model = LsiModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        chunksize=chunksize,
    )
    lsi_model.show_topics()

    ir_method = 'tfidf'  if is_tfidf else 'bow'

    lsi_model.save('saved_models/lsi_model_%s_%s' % (ir_method, num_topics))
    print('LSA for %s %s done in %.1f seconds' % (ir_method, num_topics, time.time() - start))

if __name__ == "__main__":
    arguments = [] # list of tuples
    try:
        args = sys.argv[1:]
        print(args)
        for i in range(0, len(args), 2):
            arguments.append((args[i], [int(x) for x in args[i+1].split(',')]))

        for arg in arguments:
            print(arg[0], arg[1])
            is_tfidf = True if arg[0] == 'tfidf' else False

            for num_topics in arg[1]:
                train_lsa(is_tfidf, num_topics)

    except:
        raise Exception('Arguments format: IRMethod 20,500,1000')

# Run it with command
# python lsa.py bow 10,50,100,1000,2000 tfidf 10,50,100,1000,2000