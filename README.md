# vocab
Vocab - Moz's package for generating and using n-grams. 

There are many applications where you might want to create a vocabulary. 
One example is Moz's word2gauss project (https://github.com/seomoz/word2gauss),
which computes word embeddings for the vocabulary provided to it.

This repo, vocab, will allow you to generate and use a vocabulary consisting of n-grams.
Vocabulary creation is done from the corpus that you provide.

## Dependencies
The dependency list is short: numpy and Cython (>= 0.21.1).

## Usage

### Tokenization
To create and use a vocabulary, you will need a tokenizer. If left unspecified,
a default tokenizer will be used. The default tokenizer breaks on whitespace and 
punctuation (which is removed) and keeps only tokens that contain at least one 
ASCII character. The input text must be encodable as UTF-8, and the tokens are 
lowercased.

If you supply your own tokenizer, the input will be a string and the output should
be an iterable of tokens.

### Creating a vocabulary
The vocab constructor has a few optional arguments. The *tokenizer* parameter will
allow you to specify your own tokenizer function, as described above. If you'd
prefer to use your own stopword file, you can set *stopword_file*. 

Creating n-grams from a corpus requires you to pass a tuple for each 
n-gram order: the maximum number to keep, the minimum count needed per ngram, 
and a discounting coefficient, which prevents too many ngrams consisting of very 
infrequent words to be formed. See equation 6 in this paper:

Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. 
Distributed Representations of Words and Phrases and their Compositionality. 
In Proceedings of NIPS, 2013.  (http://arxiv.org/pdf/1310.4546.pdf)

As an example, the following code creates uni, bi and trigrams. It will create
up to 75000 unigrams, 25000 bigrams, and 10000 trigrams. In this example, all
ngrams need a minimum count of 350, and 350 is the discounting coefficient. The
input, corpus, is an iterable that produces one string per document. In the example
below, the corpus is contained in a bzipped-file that contains one document per line.

```python
with BZ2File('my_corpus.bz2', 'r') as corpus:
    v = Vocabulary()
    v.create(corpus, [(75000, 350, 350), (25000, 350, 350), (10000, 350, 350)])
```

The create function also takes an optional flag, *keep_unigrams_stopwords* (True by
default) to allow you the option of not keeping stopwords in the unigram set.
For word2gauss, we want to compute the embeddings for stopwords, but for an 
application like topic modeling, you may want to exclude stopwords from the 
unigrams.

One assumption that we make when building the n-grams is that for bigrams and 
higher, valid ngrams do not start or end with stopwords. 

Ngrams are composed of unigram tokens and stored with underscores to delimit the tokens. 
For example, "statue of liberty" is stored as "statue_of_liberty".

### Saving and loading a vocabulary
You can save a vocabulary to gzipped file:
```python
v.save('my_vocab.gz')
```
Later, you can create a Vocabulary instance by loading the file:

```python
v = Vocabulary.load('my_vocab.gz')
```
The load function has optional arguments to specifiy the tokenizer (tokenizer),
the index lookup table size (table_size), and the power used to build the
index lookup table (power). The load function assumes a gzipped-file.

### Updating a vocabulary
Once a vocabulary is created, you can add n-grams to it by calling *add_ngrams*.

```python
v.add_ngrams(['iphone_6, 'samsung_galaxy_6'])
```

You can also update token counts by passing a corpus to *update_counts*, where
the corpus is an iterable that produces one string per document.

```python
v.update_counts(corpus)
```

### Using a vocabulary
Each ngram in the vocabulary is assigned an id when it is added. You can look
up an ngram by id:

```python
v.id2word(100)  
```
(example output: 'statue_of_liberty')

or an id by ngram:

```python
id = v.word2id('statue_of_liberty')
```
(example output: 100)

The function *tokenize* will return the ngrams for the input string:

```python
v.tokenize('The Statue of Liberty is in New York.')
```
example output: ['the', 'statue_of_liberty', 'is', 'in', 'new_york']

(in this example, the stopwords were retained as part of the vocabulary.)

If the input contains tokens that are not part of the vocabulary, they will be 
removed unless you set the optional parameter *remove_oov* to False. In this case,
the token "OOV" will be returned.

If you prefer the ids, you can call *tokenize_id*. The value -1 is returned for 
out-of-vocabulary tokens.

To get the size of the vocabulary, just call len:

```python
len(v)
```

### Negative sampling
Some word embedding algorithms use the idea of negative sampling, which is 
described in the paper by Mikolov et al. cited above. To enable this, we build an index 
lookup table from the vocabulary counts when you load a vocabulary from file or
create a vocabulary.

The functions, *random_id* and *random_ids* allow you to sample the vocabulary 
from this table:
 
 ```python
 id = v.random_id()        # return a random id
 ids = v.random_ids(100)   # return a list of 100 random ids
 ```
