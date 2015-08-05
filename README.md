
vocab
=====
Vocab - Moz's vocabulary package

Dependencies
============
The dependency list is short: numpy and Cython (>= 0.21.1).

Usage
=====

Create uni-, bi-, and tri-grams from a corpus:

```python
keep = [(75000, 350), (25000, 350), (10000, 350)]
vocab = Vocabulary(build_table=False)
for nkeep, min_count in keep:
    ndocs = 0
    with BZ2File('corpus.bz2', 'r') as corpus:
    for doc in corpus:
        vocab.accumulate(doc)
            ndocs += 1
    vocab.update(nkeep, min_count)
    vocab.save('my_vocab.gz')
```

Create a Vocabulary instance by loading a vocabulary file:

v = Vocabulary.load('my_vocab.gz')


