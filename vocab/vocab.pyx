
# c imports
cimport cython
cimport numpy as np

from vocab cimport *

import re
import pkgutil
import numpy as np

from gzip import GzipFile


DEFAULT_TABLE_SIZE = 10**6
DEFAULT_POWER = 0.75


re_tokenize = re.compile(r'(((?![\d|_])\w)+)', re.UNICODE)
re_keep = re.compile(r'[a-z]')
def alpha_tokenize(s):
    """
    Simple tokenizer: tokenize the the string and return a list of the tokens
    """
    if isinstance(s, unicode):
        candidates = [token[0].encode('utf-8')
            for token in re_tokenize.findall(s.lower())]
    else:
        candidates = [token[0].encode('utf-8')
            for token in re_tokenize.findall(s.decode('utf-8').lower())]
    # only keep tokens with at least one ascii character
    return [token for token in candidates if re_keep.search(token)]


cdef load_stopwords(set[string]& stopwords):
    words = pkgutil.get_data('vocab', 'stop_words.txt').strip().split()
    stopwords.clear()
    for word in words:
        stopwords.insert(word)


cdef set[string] STOP_WORDS
load_stopwords(STOP_WORDS)


cdef class Vocabulary:
    """
    Class for creating, saving, and loading a vocabulary
    """
    def __init__(self, vocab=None, tokenizer=alpha_tokenize, counts=None,
                 build_table=True, table_size=DEFAULT_TABLE_SIZE,
                 power=DEFAULT_POWER):
        self._tokenizer = tokenizer

    def __cinit__(self, vocab=None, tokenizer=None,
                  counts=None, build_table=True,
                  table_size=DEFAULT_TABLE_SIZE, power=DEFAULT_POWER):
        """
        vocab: vocabulary
        tokenizer: a tokenizer that splits strings into individual tokens
        counts: word counts, used if building the index lookup table
        build_table: if True, build the index lookup table that word2gauss
        uses for negative sampling
        table_size: index lookup table size
        power: power used in building index lookup table
        """
        cdef vocab_t v
        v.clear()

        cdef vocab_ngram_t ngram

        if vocab is not None:
            # make the unordered_map to pass into constructor
            for tokens in vocab:
                ngram.clear()
                for token, tokenid in tokens.iteritems():
                    ngram[token] = tokenid
                v.push_back(ngram)

        self._vocabptr = new Vocab(v, STOP_WORDS)
        self.counts = np.array(counts, dtype=np.uint32) if counts else None

        if build_table and counts:
            self._table = self._build_table(table_size, power)
        else:
            self._table = None

    def _build_table(self, table_size, power):
        """
        Create a table using the vocabulary tokens counts, which can be used
        for drawing random words in negative sampling routines (useful for
        word2gauss.)
        This function is called by the constructor if the build_table
        flag is set to True.

        From gensim's make_table in word2vec.py
        """
        table = np.zeros(table_size, dtype=np.uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = float(sum((count**power for count in self.counts)))
        # go through the whole table and fill it up with the word indexes
        # proportional to a word's count**power
        widx = 0
        # normalize count^0.75 by Z
        d1 = self.counts[widx]**power / train_words_pow
        vocab_size = len(self.counts)
        for tidx in xrange(table_size):
            table[tidx] = widx
            if 1.0 * tidx / table_size > d1:
                widx += 1
                d1 += self.counts[widx]**power / train_words_pow
            if widx >= vocab_size:
                widx = vocab_size - 1
        return table

    def __dealloc__(self):
        del self._vocabptr

    def __len__(self):
        return self._vocabptr.size()

    def word2id(self, string w):
        """
        Given a token string, return its id
        Raises KeyError if the string is not in the vocabulary
        """
        cdef uint32_t tokenid = self._vocabptr.get_word2id(w)
        if tokenid == -1:
            raise KeyError
        else:
            return tokenid

    def id2word(self, k):
        """
        Given an id, return the token
        Raises IndexError if the token is not in the vocabulary
        """
        cdef string token = self._vocabptr.get_id2word(<size_t>k)
        if token.size() == 0:
            raise IndexError
        else:
            return token

    def save(self, fname):
        """
        Write the vocabulary to a file
        """
        with GzipFile(fname, 'w') as fout:

            # then the n-grams
            for k in xrange(self._vocabptr.size()):
                t = self.id2word(k)
                # don't keep n-grams that end in a stop word
                # the n-gram generation code has already taken care of n-grams
                # that start with a stopword
                if '_' in t:
                    tokens = t.split('_')
                    if STOP_WORDS.find(tokens[-1]) != STOP_WORDS.end():
                        continue
                fout.write(self.id2word(k))
                fout.write('\t')
                fout.write(str(self._vocabptr.get_id2count(k)))
                fout.write('\n')

    def tokenize(self, string text, remove_oov=True):
        """
        Tokenize an input string, grouping n-grams aggressively

        text: text to tokenize
        remove_oov: if True, remove out-of-vocabulary tokens from output;
        otherwise, return 'OOV' for the out-of-vocab tokens
        """
        cdef vector[string] tokens = self._tokenizer(text)
        cdef vector[string] ret
        self._vocabptr.group_ngrams(tokens, ret, remove_oov)
        return ret

    def tokenize_ids(self, string text, remove_oov=True):
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
            ids[i] = self._vocabptr.get_word2id(token)
        return ids

    def random_id(self):
        """
        Returns a random id from the vocabulary, using the index lookup table
        """
        return self._table[np.random.randint(0, len(self._table))]

    def random_ids(self, num):
        """
        Returns an array of random ids from the vocabulary, using the index
        lookup table, which needs to have been created when the vocabulary was
        loaded.
        num: number of ids to return
        """
        idxs = np.random.randint(0, len(self._table), num)
        return self._table[idxs]

    def accumulate(self, doc):
        """
        Accumulate the n-gram counts for the document. This method tokenizes the
        input string with the tokenizer that was specified when the Vocabulary
        instance was created.
        """
        cdef vector[string] tokens = self._tokenizer(doc)
        self._vocabptr.accumulate(tokens)

    def accumulate_tokens(self, tokens):
        """
        Accumulate the n-gram counts for the list of tokens (this is useful
        when the document has already been tokenized.)
        """
        self._vocabptr.accumulate(tokens)

    def update(self, keep=100000, min_count=1):
        """
        Update the vocabulary with the accumulated counts

        keep: the number of new tokens to add to the vocabulary
        min_count: the minimum count to keep
        """
        self._vocabptr.update(keep, min_count)

    @classmethod
    def load(cls, fname, tokenizer=alpha_tokenize,
             build_table=True,
             table_size=DEFAULT_TABLE_SIZE, power=DEFAULT_POWER):
        """
        Create a Vocabulary instance, loading data from a file

        fname: name of file to load
        tokenizer: tokenizer to use with this Vocabulary instance
        build_table: if True, build the index lookup table, which is used in
        negative sampling in word2gauss
        table_size: size of lookup table
        power: power used in filling the lookup table
        """
        with GzipFile(fname, 'r') as fin:
            vocab = [{}]
            counts = []
            n = 0
            wordid = 0
            for line in fin:
                line = line.rstrip('\n')
                linearr = line.split('\t')
                token = linearr[0]
                nunderscore = len(re.findall('_', token))
                if nunderscore > n:
                    assert(n == nunderscore - 1)
                    n += 1
                    vocab.append({})
                vocab[-1][token] = wordid
                if len(linearr) > 1:
                    counts.append(int(linearr[1]))
                wordid += 1

        return cls(vocab=vocab, tokenizer=tokenizer, counts=counts,
                   build_table=build_table,
                   table_size=table_size, power=power)
