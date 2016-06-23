
import sys

import numpy as np

import pickle

from sklearn.cluster import MiniBatchKMeans

from preprocess import load_pickled

def run_kmeans(embeddings):
    kmeans = MiniBatchKMeans(n_clusters=100, batch_size=128, n_init=10, compute_labels=True)

    labels = kmeans.fit_predict(embeddings)

    return labels

def main(embedding_file, label_out):
    embeddings = load_pickled(embedding_file)

    labels = run_kmeans(embeddings)

    with open(label_out, 'wb') as f:
        pickle.dump(labels, f, protocol=-1)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
