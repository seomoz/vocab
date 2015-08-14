
vocab
=====
Vocab - Moz's vocabulary package

Dependencies
============
The dependency list is short: numpy and Cython (>= 0.21.1).

Usage
=====

Creating n-grams from a corpus requires you to pass a tuple for each 
n-gram order: the maximum number to keep, the minimum count needed per ngram, 
and a discounting coefficient, which prevents too many ngrams consisting of very 
infrequent words to be formed. See equation 6 in this paper:

Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. 
Distributed Representations of Words and Phrases and their Compositionality. 
In Proceedings of NIPS, 2013.  (http://arxiv.org/pdf/1310.4546.pdf)


As an example, the following code creates uni, bi and trigrams. It will create
up to 75000 unigrams, 25000 bigrams and 10000 trigrams. In this example, all
ngrams need a minimum count of 350, and 350 is the disounting coefficient.
```python
with BZ2File('my_corpus.bz2', 'r') as corpus:
    vocab = Vocabulary()
    vocab.create(corpus, [(75000, 350, 350), (25000, 350, 350), (10000, 350, 350)])
    vocab.save('my_vocab.gz')
```

Once a vocabulary has been saved to file, you can create a Vocabulary instance 
by loading the file:

```python
v = Vocabulary.load('my_vocab.gz')
```

