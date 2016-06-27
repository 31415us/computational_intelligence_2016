
import numpy as np

from scipy import sparse

class LabelVectorizer(object):

    def __init__(self, labels, dim=100):
        self.labels = labels
        self.dim = dim

    def vectorize(self, word_list):
        rows = []
        cols = []
        data = []

        if len(word_list) is not 0:
            norm = 1 / len(word_list)
            for w_id in word_list:
                label = self.labels[w_id]
                rows.append(0)
                cols.append(label)
                data.append(norm)

        v = sparse.coo_matrix((data, (rows, cols)), shape=(1, self.dim))
        v.sum_duplicates()

        return v.tocsr()


    #def vectorize(self, word_list):
    #    res = np.zeros(self.dim)

    #    num = len(word_list)

    #    if num == 0:
    #        return res

    #    for w_id in word_list:
    #        res[self.labels[w_id]] += 1

    #    res /= num

    #    return res
