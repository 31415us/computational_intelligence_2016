
import numpy as np

from scipy import sparse

from preprocess import MemoizedStemmer

class TfIdf(object):

    def __init__(self, vocab, infiles):
        self.idfs = idfs_from_corpus(vocab, infiles)
        self.vocab_size = len(vocab)

    def features(self, id_lists):
        vecs = []

        for l in id_lists:
            row_ids = []
            col_ids = []
            vals = []

            doc_length = len(l)

            word_counts = {}

            for w_id in l:
                if word_counts.get(w_id) is None:
                    word_counts[w_id] = 1
                else:
                    word_counts[w_id] += 1

            for w_id in word_counts:
                row_ids.append(0)
                col_ids.append(w_id)

                tfidf = (1 + np.log(word_counts[w_id] / doc_length)) * self.idfs[w_id]

                vals.append(tfidf)

            v = sparse.coo_matrix((vals, (row_ids, col_ids)), shape=(1, self.vocab_size))
            v.sum_duplicates()

            vecs.append(v)

        matrix = sparse.vstack(vecs)

        return matrix.tocsr()



def idfs_from_corpus(vocab, infiles):
    doc_counts = {}
    num_docs = 1 # assume 1 document containing every seen word

    stemmer = MemoizedStemmer()

    for filename in infiles:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                stemmed = [stemmer.stem(w) for w in line.split()]
                word_ids = [vocab.get(w) for w in stemmed if vocab.get(w) is not None]
                uniques = set(word_ids)

                num_docs += 1

                for w_id in uniques:
                    if doc_counts.get(w_id) is None:
                        doc_counts[w_id] = 1
                    else:
                        doc_counts[w_id] += 1

    idf_map = {}

    for word in vocab:
        w_id = vocab[word]

        try:
            idf = num_docs / (doc_counts[w_id] + 1) # assumy dummy document containing each word once
        except KeyError:
            idf = num_docs

        idf_map[w_id] = np.log(idf)

    return idf_map
