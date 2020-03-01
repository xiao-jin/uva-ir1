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
from gensim.models import LdaModel
from gensim.models import ldamulticore

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

def train_lda(is_tfidf, num_topics):
    # Create corpus
    print('Create corpus')
    corpus = doc_processor.create_corpus(dictionary, doc_list, is_tfidf)

    # Set training parameters.
    num_topics = num_topics
    chunksize = 20000
    # passes = 20
    # iterations = 400
    eval_every = None

    print('Start LDI training')
    start = time.time()
    id2word = dictionary.id2token

    lda_model = LdaModel(
        corpus=corpus,
        # id2word=id2word,
        chunksize=chunksize,
        # alpha='auto',
        # eta='auto',
        num_topics=num_topics,
        # passes=passes,
        # iterations=iterations,
        eval_every=eval_every
    )
    
    ir_method = 'tfidf'  if is_tfidf else 'bow'
    lda_model.save('saved_models/lda_model_%s_%s' % (ir_method, num_topics))
    print('LDA for %s %s done in %.1f seconds' % (ir_method, num_topics, time.time() - start))

if __name__ == "__main__":
    arguments = [] # list of tuples
    try:
        args = sys.argv[1:]
        print(args)
        for i in range(0, len(args), 2):
            arguments.append((args[i], [int(x) for x in args[i+1].split(',')]))
    except:
        raise Exception('Arguments format: IRMethod 20,500,1000')

    for arg in arguments:
        print(arg[0], arg[1])
        is_tfidf = True if arg[0] == 'tfidf' else False

        for num_topics in arg[1]:
            train_lda(is_tfidf, num_topics)


    
# Run it with command
# python train_lda.py tfidf 10,50,100,500,1000,2000