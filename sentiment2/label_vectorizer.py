
import numpy as np

class LabelVectorizer(object):

    def __init__(self, labels, dim=100):
        self.labels = labels
        self.dim = dim

    def vectorize(self, word_list):
        res = np.zeros(self.dim)

        num = len(word_list)

        if num == 0:
            return res

        for w_id in word_list:
            res[self.labels[w_id]] += 1

        res /= num

        return res
