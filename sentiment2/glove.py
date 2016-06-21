
import numpy as np

import pickle

from preprocess import load_pickled

class GloVe(object):

    def __init__(self, cooc, embedding_dim=128):
        self.cooc = cooc.tocsc()
        self.embedding_dim = embedding_dim
        self.xs = np.random.normal(size=(self.cooc.shape[0], self.embedding_dim))
        self.ys = np.random.normal(size=(self.cooc.shape[1], self.embedding_dim))

    def training_run(self, alpha=0.75, eta=0.001, nmax=100):
        training_entries = np.transpose(self.cooc.nonzero())
        np.random.shuffle(training_entries)

        for i in range(0, training_entries.shape[0]):
            ix = training_entries[i, 0]
            jy = training_entries[i, 1]

            n = self.cooc[ix, jy]

            logn = np.log(n)

            fn = min(1.0, (n / nmax) ** alpha)

            x, y = self.xs[ix, :], self.ys[jy, :]

            scale = 2 * eta * fn * (logn - np.dot(x, y))

            self.xs[ix, :] += scale * y
            self.ys[jy, :] += scale * x

    def save(self, out_file):
        with open(out_file, 'wb') as f:
            pickle.dump(self.xs, f, protocol=-1)


def main():

    cooc = load_pickled('cooc.dat')

    glove = GloVe(cooc)

    for epoch_num in range(0, 5):
        print("start epoch " + str(epoch_num))
        glove.training_run()

    print("finished")

    glove.save('embeddings.dat')



if __name__ == "__main__":
    main()
