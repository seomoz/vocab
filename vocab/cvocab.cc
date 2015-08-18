
#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <set>
#include <algorithm>
#include <utility>

typedef std::unordered_map<std::string, uint32_t> vocab_ngram_t;
typedef std::vector<vocab_ngram_t> vocab_t;
typedef std::set<std::string> stopwords_t;
typedef std::unordered_map<uint64_t, uint32_t> bigram_t;
typedef std::pair<std::string, uint32_t> ngram_count_t;
typedef std::vector<std::pair<ngram_count_t, double> > ngram_pmi_t;

void u32_to_u64(uint32_t x1, uint32_t x2, uint64_t & y)
{
    // pack two uint32_t into one uint64_t
    y = ((uint64_t)x2) << 32 | x1;
}

void u64_to_u32(uint64_t y, uint32_t & x1, uint32_t & x2)
{
    // pack one uint64_t into two uint32_t
    x1 = y & 0xFFFFFFFFU;
    x2 = y >> 32;
}

class Vocab
{
    public:
        Vocab(vocab_t vocab, stopwords_t stopwords);
        ~Vocab();


        // for creating the vocab
        void accumulate(std::vector<std::string> &);
        void update(uint32_t, uint32_t, uint32_t);
          // save is called when the caller is done creating the vocabulary
        void save(bool keep_unigram_stopwords);

        // take a doc as a list of tokens and group the n-grams
        // aggressively
        void group_ngrams(std::vector<std::string> & tokens,
            std::vector<std::string> & ret, bool remove_oov);

        uint32_t get_word2id(std::string & token);
        std::string get_id2word(std::size_t);
        uint32_t get_id2count(std::size_t);

        // total words in the vocab
        uint32_t size(void);

        void add_ngram(std::string s, size_t order);


    private:
        vocab_t vocab;
        std::vector<std::string> id2word;
        int max_ngram;
        std::vector<std::uint32_t> id2count;

        // stopword set
        stopwords_t stopwords;

        // for accumulating vocab and finding n-grams
        vocab_ngram_t unigrams_map;
        std::vector<uint32_t> unigrams;
        bigram_t bigrams;

        // disable some default constructors
        Vocab();
        Vocab& operator= (const Vocab& other);
        Vocab(const Vocab& other);

        void populate_id2word(uint32_t total_words);
};


Vocab::Vocab(vocab_t vocab, stopwords_t stopwords)
{
    this->vocab = vocab;
    max_ngram = vocab.size();
    this->stopwords = stopwords;

    // make id2word
    // get the total number of words in the vocab
    uint32_t total_words = 0;
    for (vocab_t::iterator it_vocab = vocab.begin(); it_vocab != vocab.end();
         ++it_vocab)
    {
        total_words += it_vocab->size();
    }

    populate_id2word(total_words);

    // initialize the attributes for accumulating n-grams
    unigrams_map.clear();
}

Vocab::~Vocab() {}

uint32_t Vocab::size(void)
{
    return id2word.size();
}

void Vocab::add_ngram(std::string s, size_t order)
{
    if (order >= vocab.size())
    {
        // ngrams could come out of order, so make sure vocab is large enough
        int num = order - vocab.size() + 1;
        for (int i = 0; i < num; ++i)
        {
            vocab.push_back(vocab_ngram_t());
        }
    }
    uint32_t id = id2word.size();
    vocab[order][s] = id;
    id2word.push_back(s);
    id2count.push_back(0);
    max_ngram = vocab.size();
}

void Vocab::populate_id2word(uint32_t total_words)
{
    id2word.resize(total_words);

    // now populate
    for (vocab_t::iterator it_vocab = vocab.begin(); it_vocab != vocab.end();
         ++it_vocab)
    {
        for (vocab_ngram_t::iterator it = it_vocab->begin();
             it != it_vocab->end(); ++it)
        {
            id2word[it->second] = it->first;
        }
    }
}

void Vocab::group_ngrams(std::vector<std::string> & tokens,
            std::vector<std::string> & ret, bool remove_oov)
{
    // want to return a list of grouped n-grams where each n-gram
    // in the vocab is replaced with the individual tokens with underscores
    // we replace the longest n-grams first, then work to the shortest

    // strategy: replace n-grams in place with the first word, then
    // set the rest to empty strings
    // the last pass to check for unigrams will remove them

    if (max_ngram == 0)
    {
        // no vocab
        ret.clear();
        return;
    }

    int ntokens = tokens.size();
    for (std::size_t ngram_length = std::min(max_ngram, ntokens);
        ngram_length > 1; --ngram_length)
    {
        // don't need to check unigrams
        std::size_t ind = ngram_length - 1;

        for (std::size_t i = 0; i < ntokens - ind; ++i)
        {
            std::string ngram = tokens[i];
            if (ngram == "")
                // this token already checked and removed
                continue;
            for (std::size_t ii = i + 1; ii < i + ngram_length; ++ii)
            {
                ngram += "_";
                ngram += tokens[ii];
            }
            vocab_ngram_t::iterator got = vocab[ind].find(ngram);
            if (got != vocab[ind].end())
            {
                // found an n-gram
                tokens[i] = ngram;
                for (std::size_t j = i + 1; j < i + ngram_length; ++j)
                    tokens[j] = "";
            }
        }
    }

    // remove all empty strings
    // If remove_oov is true, remove the out-of-vocabulary unigrams; otherwise,
    // add "OOV" to the return
    ret.clear();
    for (std::vector<std::string>::iterator it = tokens.begin();
         it != tokens.end(); ++it)
    {
        vocab_ngram_t::iterator got_unigram = vocab[0].find(*it);
        bool is_unigram = got_unigram != vocab[0].end();
        std::size_t got_ngram = it->find("_");
        bool is_ngram = got_ngram != std::string::npos;
        if (*it != "")
        {
            if (is_unigram || is_ngram)
                ret.push_back(*it);
            else if (!remove_oov)
                ret.push_back("OOV");
        }
    }
}

uint32_t Vocab::get_word2id(std::string & word)
{
    // get the token id
    for (std::size_t k = 0; k < vocab.size(); ++k)
    {
        vocab_ngram_t::iterator got = vocab[k].find(word);
        if (got != vocab[k].end())
            return got->second;
    }

    // otherwise return -1 (note: will be converted to max unsigned int32)
    return -1;
}

std::string Vocab::get_id2word(std::size_t k)
{
    // get the word corresponding to this id
    if (k < id2word.size())
        return id2word[k];
    else
        return "";  // return "" if id doesn't exist
}

uint32_t Vocab::get_id2count(std::size_t k)
{
    // get the word count for this id
    if (k < id2count.size())
        return id2count[k];
    else
        return 0;   // return 0 if the id doesn't exist
}

void Vocab::accumulate(std::vector<std::string> & tokens)
{
    // accumulate n-gram counts for the doc
    if (max_ngram == 0)
    {
        // just need the unigrams for now
        for (std::vector<std::string>::iterator  it = tokens.begin();
             it != tokens.end(); ++it)
        {
            unigrams_map[*it] += 1;
        }
    }
    else if (tokens.size() >= max_ngram + 1)
    {
        // we've already calculated some lower order ngrams
        // need to accumulate counts for higher order ones
        // only keep unigram counts for words of the next lower
        // order n-gram + the stop words and bi-gram counts
        // for bi-grams starting with the next lower order and
        // followed by a unigram word or a stop word

        // update unigram counts first
        for (std::vector<std::string>::iterator it = tokens.begin();
             it != tokens.end(); ++it)
        {
            // check whether it is a unigram word
            vocab_ngram_t::iterator it_vocab = vocab[0].find(*it);
            if (it_vocab != vocab[0].end())
                unigrams[it_vocab->second] += 1;
        }

        // now update the n-gram counts
        std::size_t n = max_ngram + 1;
        for (std::size_t k = 0; k < tokens.size() - (n - 1); ++k)
        {
            std::string ngram = tokens[k];
            // we don't want n-grams that start with stopwords
            if (stopwords.find(ngram) != stopwords.end())
                continue;

            for (std::size_t i = k + 1; i < k + n - 1; ++i)
                ngram += "_" + tokens[i];
            vocab_ngram_t::iterator it_vocab = vocab[max_ngram - 1].find(ngram);
            if (it_vocab != vocab[max_ngram - 1].end())
            {
                // a valid start n-gram
                // check if the last token is valid
                uint32_t ngramid = it_vocab->second;
                std::string next_token = tokens[k + n - 1];
                uint32_t next_tokenid;
                it_vocab = vocab[0].find(next_token);
                if (it_vocab != vocab[0].end())
                {
                    next_tokenid = it_vocab->second;

                    // we have a valid ngram!  update the counts
                    if (n > 2)
                        unigrams[ngramid] += 1;
                    uint64_t bigram;
                    u32_to_u64(ngramid, next_tokenid, bigram);
                    std::pair<bigram_t::iterator, bool> inserted =
                        bigrams.insert(std::make_pair(bigram, 1));
                    if (!inserted.second)
                        inserted.first->second += 1;
                }
            }
        }
    }
}


template <class T1, class T2>
struct OrderByDescendingSecondComponent {
    bool operator()(
        const std::pair<T1, T2> &a, const std::pair<T1, T2> &b) const {
        return b.second < a.second;
    }
};


void Vocab::update(uint32_t num_to_keep, uint32_t min_count, uint32_t delta)
{
    if (max_ngram == 0)
    {
        // it's the first pass to set the unigrams
        std::vector<std::pair<std::string, uint32_t> > chosen;
        OrderByDescendingSecondComponent<std::string, uint32_t> comparator;
        std::make_heap(chosen.begin(), chosen.end(), comparator);

        uint32_t words_kept = 0;
        for (vocab_ngram_t::iterator it = unigrams_map.begin();
             it != unigrams_map.end(); ++it)
        {
            if (it->second >= min_count)
            {
                // push onto the heap and pop smallest
                words_kept += 1;
                chosen.push_back(*it);
                std::push_heap(chosen.begin(), chosen.end(), comparator);
                if (words_kept > num_to_keep) {
                    std::pop_heap(chosen.begin(), chosen.end(), comparator);
                    chosen.pop_back();
                }
            }
        }

        // now that we have the chosen unigrams we can set the vocab
        vocab.push_back(vocab_ngram_t());

        for (std::size_t k = 0; k < chosen.size(); ++k)
        {
            vocab[0][chosen[k].first] = k;
            id2word.push_back(chosen[k].first);
            id2count.push_back(chosen[k].second);
        }
    }
    else
    {
        // update the higher order ngrams
        // PMI (pointwise mutual information) approach:
        // want to keep ones with the highest p(x, y) / (p(x) * p(y))
        // p(x) = count(x) / total unigrams
        // p(x, y) = count(x, y) / total bigrams
        // so p(x, y) / (p(x) * p(y)) = count(x, y) / (count(x) * count(y)) * fac
        // where fac = total unigrams **2 / total bigrams and is the same
        // for each word so we can ignore it
        // We use this approach with one modification that is described here:
        // Equation 6 from:
        // Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and
        // Jeffrey Dean. Distributed Representations of Words and Phrases and
        // their Compositionality. In Proceedings of NIPS, 2013.
        // delta = discounting coefficient, used to prevent too many phrases
        // consisting of very infrequent words to be formed.
        ngram_pmi_t chosen;
        OrderByDescendingSecondComponent<ngram_count_t, double> comparator;
        std::make_heap(chosen.begin(), chosen.end(), comparator);

        uint32_t words_kept = 0;
        uint32_t ntokens_vocab = id2word.size();

        for (bigram_t::iterator it = bigrams.begin(); it != bigrams.end(); ++it)
        {
            if (it->second >= min_count)
            {
                uint32_t first_id, second_id;
                u64_to_u32(it->first, first_id, second_id);

                std::string ngram = id2word[first_id] + "_" +
                    id2word[second_id];

                double denom = unigrams[first_id] * unigrams[second_id];
                double pmi = (it->second - delta) / denom;

                // push onto heap and pop smallest
                words_kept += 1;
                chosen.push_back(std::make_pair(
                    std::make_pair(ngram, it->second), pmi));
                std::push_heap(chosen.begin(), chosen.end(), comparator);
                if (words_kept > num_to_keep)
                {
                    std::pop_heap(chosen.begin(), chosen.end(), comparator);
                    chosen.pop_back();
                }
            }
        }

        // now add the bigrams to the vocab
        vocab.push_back(vocab_ngram_t());

        for (ngram_pmi_t::iterator it_chosen = chosen.begin();
             it_chosen != chosen.end(); ++it_chosen)
        {
            vocab[max_ngram][it_chosen->first.first] = ntokens_vocab;
            id2word.push_back(it_chosen->first.first);
            id2count.push_back(it_chosen->first.second);
            ntokens_vocab += 1;
        }
    }

    // reset the internal counters
    unigrams_map.clear();
    unigrams.clear();
    bigrams.clear();
    unigrams.resize(id2word.size(), 0);
    max_ngram = vocab.size();
}

// Hardens the vocabulary; once save is called, no higher ngrams can be
// computed.
// Keeping stopwords in the vocabulary may not be desirable for some
// applications, so the keep_unigram_stopwords flag allows the caller to
// control this behavior
void Vocab::save(bool keep_unigram_stopwords)
{
    uint32_t next_id;
    std::size_t n;
    if (keep_unigram_stopwords)
    {
        // if keeping unigram stopwords, start with bigrams and higher
        next_id = vocab[0].size();
        n = 1;
    }
    else
    {
        // otherwise, we also examine the unigrams and exclude stopwords
        next_id = 0;
        n = 0;
    }

    // Loop over the ngram orders...
    for (std::size_t k = n; k < vocab.size(); ++k)
    {
        vocab_ngram_t to_keep;
        // Loop over the items...
        for (vocab_ngram_t::iterator it = vocab[k].begin();
             it != vocab[k].end(); ++it)
         {
            // Check last token, see if it's a stopword
            std::string ngram = it->first;
            std::size_t found = ngram.find_last_of("_");
            std::string last_token = ngram.substr(found+1);
            if (stopwords.find(last_token) == stopwords.end())
            {
                // Last token's not a stopword, so keep this ngram
                to_keep[ngram] = next_id;
                next_id++;
            }
         }
         vocab[k] = to_keep;
     }

    // id2word map has to be re-populated
    populate_id2word(next_id);
}
