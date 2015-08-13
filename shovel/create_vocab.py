
"""
Shovel task for creating n-grams for a corpus
"""

import logging

from bz2 import BZ2File
from vocab import Vocabulary
from shovel import task

logger = logging.getLogger('prepare')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)


@task
def ngrams(prefix):
    """
    Find n-grams and make a vocabulary from the parsed corpus
    """
    with BZ2File(prefix + 'corpus.bz2', 'r') as corpus:
        vocab = Vocabulary(build_table=False)
        vocab.create(corpus, [(75000, 350), (25000, 350), (10000, 350)])
    vocab.save(prefix + 'vocab.gz')


@task
def count(prefix):
    """
    Count the number of tokens in the corpus
    """
    vocab = Vocabulary.load(prefix + 'vocab.gz', build_table=False)
    total = 0
    ndocs = 0
    with BZ2File(prefix + 'corpus.bz2', 'r') as corpus:
        for doc in corpus:
            tokens = vocab.tokenize(doc)
            total += len(tokens)
            ndocs += 1
            if ndocs % 10000 == 0:
                logger.info("Processed %s docs." % ndocs)
    logger.info("Total of %s tokens in corpus" % total)
