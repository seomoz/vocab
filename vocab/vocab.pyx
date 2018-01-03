# c imports
cimport cython
cimport numpy as np

from vocab cimport *

import re
import pkgutil
import numpy as np

from gzip import open as gzopen
from io import TextIOWrapper, BufferedReader, BufferedWriter


DEFAULT_TABLE_SIZE = 10**6
DEFAULT_POWER = 0.75

LARGEST_UINT32 = 4294967295

re_tokenize = re.compile(r'(((?![\d|_])\w)+)', re.UNICODE)
re_keep = re.compile(r'[a-z]')
def alpha_tokenize(s):
    """
    Simple tokenizer: tokenize the the string and return a list of the tokens
    """
    if isinstance(s, unicode):
        candidates = [token[0]
            for token in re_tokenize.findall(s.lower())]
    else:
        candidates = [token[0]
            for token in re_tokenize.findall(s.decode('utf-8').lower())]
    # only keep tokens with at least one ascii character
	#TODO: is this what you really want?
    return [token for token in candidates if re_keep.search(token)]


def load_stopwords(stop_file=None):
    '''
    stop_file is None (use default list), or string (with filename) or
        iterable
    '''
    stopwords = set()
    if stop_file is None:
        words = pkgutil.get_data('vocab', 'stop_words.txt').strip().split()
    elif isinstance(stop_file, basestring):
        # a file name
        with open(stop_file, 'r') as f:
            words = [line.rstrip() for line in f]
    else:
        # assume to be iterable
        words = stop_file
    for word in words:
        stopwords.add(word)
    return stopwords


class IndexLookupTable(object):
    def __init__(self, counts, table_size, power):
        self._table_size = table_size
        self._power = power
        if counts is None:
            self._table = None
        else:
            self._table = IndexLookupTable._build(counts, table_size, power)

    def update_table(self, counts):
        """ Update the table with new counts"""
        # just build a new table with the counts
        self._table = IndexLookupTable._build(counts,
                                              self._table_size, self._power)

    @staticmethod
    def _build(counts, table_size, power):
        """
        Create a table using the vocabulary tokens counts, which can be used
        for drawing random words in negative sampling routines (useful for
        word2gauss.)

        From gensim's make_table in word2vec.py
        """
        table = np.zeros(table_size, dtype=np.uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = float(sum((count**power for count in counts)))
        if train_words_pow == 0.0:
            return None

        # go through the whole table and fill it up with the word indexes
        # proportional to a word's count**power
        widx = 0
        # normalize count^0.75 by Z
        d1 = counts[widx]**power / train_words_pow
        vocab_size = len(counts)
        for tidx in xrange(table_size):
            table[tidx] = widx
            if 1.0 * tidx / table_size > d1:
                widx += 1
                d1 += counts[widx]**power / train_words_pow
            if widx >= vocab_size:
                widx = vocab_size - 1
        return table

    def random_id(self):
        return self._table[np.random.randint(0, len(self._table))]

    def random_ids(self, num):
        idxs = np.random.randint(0, len(self._table), num)
        return self._table[idxs]


cdef class Vocabulary:
    """
    Class for creating, saving, and loading a vocabulary
    """
    def __init__(self, vocab=None, tokenizer=alpha_tokenize,
                 stopword_file=None, counts=None,
                 table_size=DEFAULT_TABLE_SIZE, power=DEFAULT_POWER):
        self._tokenizer = tokenizer


    def __cinit__(self, vocab=None, tokenizer=None, stopword_file=None,
                  counts=None,
                  table_size=DEFAULT_TABLE_SIZE, power=DEFAULT_POWER):
        """
        vocab: vocabulary
        tokenizer: a tokenizer that splits strings into individual tokens
        stopword_file: file containing stopwords. If None, the default
            stopword list is used.  If an iterable then gives
            the list of stop words
        counts: word counts
        table_size: index lookup table size
        power: power used in building index lookup table
        """
        cdef vocab_t v
        v.clear()

        cdef vocab_ngram_t ngram

        if vocab:
            # make the unordered_map to pass into constructor
            for tokens in vocab:
                ngram.clear()
                for token, tokenid in tokens.iteritems():
                    ngram[token.encode('utf-8') if isinstance(token, unicode) else token] = tokenid
                v.push_back(ngram)

        self._stopwords = load_stopwords(stopword_file)
        self._vocabptr = new Vocab(v, self._stopwords)
        self.counts = np.array(counts, dtype=np.uint32) if counts else None
        self._lookup_table = IndexLookupTable(counts, table_size, power)

    def __dealloc__(self):
        del self._vocabptr

    def __len__(self):
        return self._vocabptr.size()

    def create(self, corpus, ngram_counts, keep_unigram_stopwords=True):
        """
        Create the vocabulary from the given corpus

        corpus: an iterable that produces documents (as strings) from a corpus
        ngram_counts: a list of tuples, one per ngram order that specifies the
        maximum number of tokens to keep, the minimum count required
        for a token, and a discounting coefficient (which prevents too many
        ngrams consisting of very infrequent words to be formed)
        keep_unigram_stopwords: if True (default), keep the unigrams that are
        stopwords. If False, remove the stopwords from the vocabulary
        """
        for nkeep, min_count, delta in ngram_counts:
            for doc in corpus:
                self._vocabptr.accumulate([token.encode('utf-8') for token in self._tokenizer(doc)])
            self._vocabptr.update(nkeep, min_count, delta)
            # if corpus is a file iterator, we need to seek to the
            # beginning in order to iterate over it again
            if hasattr(corpus, 'seek'):
                corpus.seek(0)
        self._vocabptr.save(keep_unigram_stopwords)
        vlen = self._vocabptr.size()
        self.counts = np.zeros(vlen, dtype=np.uint32)
        for k in xrange(vlen):
            self.counts[k] = self._vocabptr.get_id2count(k)
        self._lookup_table.update_table(self.counts)

    def add_ngrams(self, ngrams, exclude_stopwords=False):
        """
        Add the list of ngrams to the vocabulary. This is meant for when we
        want to update the vocabulary with new items.

        ngram: iterable of ngrams to add
        """
        count = 0
        for ngram in ngrams:
		#TODO: just automatically encode ngram?
            if exclude_stopwords and ngram in self._stopwords:
                continue
            # avoid duplicates in the vocabulary
            tokenid = self._vocabptr.get_word2id(ngram.encode('utf-8'))
            if tokenid == LARGEST_UINT32:
                order = ngram.count('_')
                self._vocabptr.add_ngram(ngram.encode('utf-8'), order)
                count += 1
        if self.counts is None:
             self.counts = np.zeros(self._vocabptr.size(), dtype=np.uint32)
        else:
            tmp = np.zeros(count, dtype=np.uint32)
            self.counts = np.concatenate((self.counts, tmp), axis=0)

    def is_stopword(self, word):
        return word in self._stopwords

    def update_counts(self, corpus):
        """
        Update the counts of the current vocabulary with the given corpus
        """
        for doc in corpus:
            ids = self.tokenize_ids(doc, remove_oov=True)
            self.counts[ids] += 1
        self._lookup_table.update_table(self.counts)

    def word2id(self, w):
        """
        Given a token string, return its id
        Raises KeyError if the string is not in the vocabulary
        """
        cdef uint32_t tokenid = self._vocabptr.get_word2id(w.encode('utf-8'))
        if tokenid == LARGEST_UINT32:
            raise KeyError
        else:
            return tokenid

    def id2word(self, k):
        """
        Given an id, return the token
        Raises IndexError if the token is not in the vocabulary
        """
        token = self._vocabptr.get_id2word(<size_t>k)
        if token.size() == 0:
            raise IndexError
        else:
            return token.decode('utf-8')

    def save(self, fname):
        """
        Write the vocabulary to a file
        """
        with TextIOWrapper(BufferedWriter(gzopen(fname, 'w')), encoding='utf-8') as fout:
            #equivalent to using mode="wt" but works in both Python 2 and 3
            for k in xrange(self._vocabptr.size()):
                fout.write(self.id2word(k))
                fout.write(u'\t')
                fout.write(unicode(self._vocabptr.get_id2count(k)))
                fout.write(u'\n')

    def tokenize(self, text, remove_oov=True):
        """
        Tokenize an input string, grouping n-grams aggressively

        text: text to tokenize
        remove_oov: if True, remove out-of-vocabulary tokens from output;
        otherwise, return 'OOV' for the out-of-vocab tokens
        """
        cdef vector[string] tokens = [token.encode('utf-8') for token in self._tokenizer(text)]
        cdef vector[string] ret
        self._vocabptr.group_ngrams(tokens, ret, remove_oov)
        return [token.decode('utf-8') for token in ret]

    def tokenize_ids(self, text, remove_oov=True):
        """
        Tokenize an input string, grouping n-grams aggressively. This function
        returns the token ids.

        text: text to tokenize
        remove_oov: if True, remove out-of-vocabulary tokens from output;
        otherwise, return 'OOV' for the out-of-vocab tokens
        """
        tokens = self.tokenize(text, remove_oov)
        ids = np.zeros(len(tokens), dtype=np.uint32)
        for i, token in enumerate(tokens):
            ids[i] = self._vocabptr.get_word2id(token.encode('utf-8'))
        return ids

    def random_id(self):
        """
        Returns a random id from the vocabulary, using the index lookup table
        """
        return self._lookup_table.random_id()

    def random_ids(self, num):
        """
        Returns an array of random ids from the vocabulary, using the index
        lookup table

        num: number of ids to return
        """
        return self._lookup_table.random_ids(num)

    @classmethod
    def load(cls, fname, tokenizer=alpha_tokenize, stopwords_file=None,
             table_size=DEFAULT_TABLE_SIZE, power=DEFAULT_POWER):
        """
        Create a Vocabulary instance, loading data from a file

        fname: name of file to load
        tokenizer: tokenizer to use with this Vocabulary instance
        table_size: size of the index lookup table (which is used in negative
        sampling in word2gauss)
        power: power used in filling the index lookup table
        """
        with TextIOWrapper(BufferedReader(gzopen(fname, 'r')), encoding='utf-8') as fin:
            #equivalent to using mode="wt" but works in both Python 2 and 3
            vocab = []
            counts = []
            wordid = 0
            for line in fin:
                line = line.rstrip('\n')
                linearr = line.split('\t')
                token = linearr[0]
                # number of underscores tells us what kind of ngram this is
                # (and index into vocab list)
                idx = token.count('_')
                # num: number of {} to add to list
                num = idx - len(vocab) + 1
                for i in xrange(num):
                    vocab.append({})
                vocab[idx][token] = wordid
                if len(linearr) > 1:
                    counts.append(int(linearr[1]))
                wordid += 1

        return cls(vocab=vocab, tokenizer=tokenizer, counts=counts,
                   stopword_file=stopwords_file,
                   table_size=table_size, power=power)
