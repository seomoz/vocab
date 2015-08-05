
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libc.stdint cimport uint32_t
from libcpp cimport bool

cimport numpy as np

ctypedef unordered_map[string, uint32_t] vocab_ngram_t
ctypedef vector[vocab_ngram_t] vocab_t
ctypedef map[string, uint32_t] stop_t

# wrapper for the C++ Vocab class
cdef extern from "cvocab.cc":
    cdef cppclass Vocab:
        Vocab(vocab_t) except +
        Vocab() except +
        void group_ngrams(vector[string] &, vector[string] &, bool)
        void accumulate(vector[string] &, stop_t &)
        void update(uint32_t, uint32_t, stop_t &, map[uint32_t, string] &)
        uint32_t get_word2id(string)
        string get_id2word(size_t)
        uint32_t get_swid2count(size_t)
        uint32_t get_id2count(size_t)
        uint32_t size()

cdef class Stopwords:
    cdef map[string, uint32_t] word2id
    cdef map[uint32_t, string] id2word
    cdef public uint32_t min_id

cdef class Vocabulary:
    cdef Vocab *_vocabptr
    cdef np.ndarray _table
    cdef object _tokenizer
    cpdef public np.ndarray counts
    cdef public Stopwords stopwords
