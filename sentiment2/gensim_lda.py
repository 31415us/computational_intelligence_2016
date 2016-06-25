
import sys 

import gensim

from gensim.models.ldamodel import LdaModel

import numpy as np

from preprocess import MemoizedStemmer, load_pickled

class TweetCorpus(gensim.corpora.textcorpus.TextCorpus):

    def __init__(self, infile, vocab):
        super(TweetCorpus, self).__init__()
        self.infile = infile
        self.stemmer = MemoizedStemmer()
        self.vocab = vocab

    def get_texts(self):
        with open(self.infile, 'r', encoding='utf-8') as f:
            for line in f:
                stemmed = [self.stemmer.stem(w) for w in line.split()]
                filter_vocab = [w for w in stemmed if self.vocab.get(w) is not None]
                yield filter_vocab

    def __iter__(self):
        for txt in self.get_texts():
            w_ids = [self.vocab[w] for w in txt]
            yield list_to_bow(w_ids)

def list_to_bow(word_ids):
    d = {}

    for w_id in word_ids:
        if d.get(w_id) is None:
            d[w_id] = 1
        else:
            d[w_id] += 1

    return list(d.items())

class LdaLoader(object):

    def __init__(self, lda_file, num_topics):
        self.lda = LdaModel.load(lda_file)
        self.num_topics = num_topics

    def topic_vector(self, word_ids):
        topic_mix = self.lda[list_to_bow(word_ids)]

        res = np.zeros(self.num_topics)

        for ix, val in topic_mix:
            res[ix] = val

        return res

def main(vocab_file, inv_vocab_file, infiles):
    vocab = load_pickled(vocab_file)
    inv_vocab = load_pickled(inv_vocab_file)

    lda = LdaModel(id2word=inv_vocab, num_topics=200)

    for f in infiles:
        tc = TweetCorpus(f, vocab)
        lda.update(tc)

    lda.save('topics.lda')

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
