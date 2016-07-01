
import numpy as np

import pickle

from sklearn.cluster import MiniBatchKMeans

from preprocess import load_pickled

def run_kmeans(embeddings, kmeans_random_seed):
    kmeans = MiniBatchKMeans(
            n_clusters=100,
            batch_size=128,
            n_init=10,
            compute_labels=True,
            random_state=kmeans_random_seed)

    labels = kmeans.fit_predict(embeddings)

    return labels

def train_kmeans(embedding_file, label_out, kmeans_random_seed):
    embeddings = load_pickled(embedding_file)

    labels = run_kmeans(embeddings, kmeans_random_seed)

    with open(label_out, 'wb') as f:
        pickle.dump(labels, f, protocol=-1)

if __name__ == "__main__":
    pass
