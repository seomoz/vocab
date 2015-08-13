
vocab
=====
Vocab - Moz's vocabulary package

Dependencies
============
The dependency list is short: numpy and Cython (>= 0.21.1).

Usage
=====

Create uni, bi, and trigrams from a corpus:

```python
with BZ2File('my_corpus.bz2', 'r') as corpus:
    vocab = Vocabulary(build_table=False)
    vocab.create(corpus, [(75000, 350), (25000, 350), (10000, 350)])
    vocab.save('my_vocab.gz')
```

Create a Vocabulary instance by loading a vocabulary file:

```python
v = Vocabulary.load('my_vocab.gz')
```

