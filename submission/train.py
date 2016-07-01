
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier

from scipy import sparse

from tfidf import TfIdf
from label_vectorizer import LabelVectorizer

from preprocess import MemoizedStemmer, load_pickled

class Sample(object):
    def __init__(self, tfidf, glove, label):
        self.tfidf = tfidf
        self.glove = glove
        self.label = label

    def kernel(self, b):
        tfidfdot = np.dot(self.tfidf.todense(), b.tfidf.todense().T)
        glovedot = np.dot(self.glove.todense(), b.glove.todense().T)

        return tfidfdot + glovedot

class TrainingSampleProvider(object):

    def __init__(self, filename, label, vocab, tfidf, label_vectorizer, stemmer):
        self.filename = filename
        self.vocab = vocab
        self.tfidf = tfidf
        self.label_vectorizer = label_vectorizer
        self.stemmer = stemmer
        self.label = label

    def tokenize(self, line):
        return line.split()

    def samples(self):
        with open(self.filename, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = self.tokenize(line)
                stemmed = [self.stemmer.stem(w) for w in tokens]
                filtered = [self.vocab.get(w) for w in stemmed if self.vocab.get(w) is not None]

                tfidf_vec = self.tfidf.features([filtered])
                label_vec = self.label_vectorizer.vectorize(filtered)

                yield Sample(tfidf_vec, label_vec, self.label)

class ValidationSampleProvider(TrainingSampleProvider):
    def __init__(self, filename, label, vocab, tfidf, label_vectorizer, stemmer):
        super().__init__(filename, label, vocab, tfidf, label_vectorizer, stemmer)

    def tokenize(self, line):
        split_index = line.split(',')
        rejoin = ' '.join(split_index[1:])

        return rejoin.split()

class SampleMerger(object):
    
    def __init__(self, pos_samples, neg_samples):
        self.pos_samples = pos_samples.samples()
        self.neg_samples = neg_samples.samples()

    def samples(self):

        while True:
            if self.pos_samples is None and self.neg_samples is None:
                raise StopIteration
            elif self.pos_samples is None:
                try:
                    sample = next(self.neg_samples)
                except StopIteration:
                    self.neg_samples = None
                    continue
            elif self.neg_samples is None:
                try:
                    sample = next(self.pos_samples)
                except StopIteration:
                    self.pos_samples = None
                    continue
            else:
                if np.random.uniform() < 0.5:
                    try:
                        sample = next(self.pos_samples)
                    except StopIteration:
                        self.pos_samples = None
                        continue
                else:
                    try:
                        sample = next(self.neg_samples)
                    except StopIteration:
                        self.neg_samples = None
                        continue

            yield sample

def samples_to_matrix(samples):
    X_list = []
    y_list = []

    for sample in samples:
        feature_vec = sparse.hstack([sample.tfidf, sample.glove])
        X_list.append(feature_vec)
        y_list.append(sample.label)

    X = sparse.vstack(X_list)
    y = np.array(y_list)

    return X, y

def batch(gen, batch_size):
    exhausted = False
    while not exhausted:
        res = []
        while len(res) < batch_size:
            try:
                res.append(next(gen))
            except StopIteration:
                exhausted = True
                break

        yield res


def train_setup(vocab_file, pos_file, neg_file, cluster_labels_file, validation_file):
    vocab = load_pickled(vocab_file)
    tfidf = TfIdf(vocab, [pos_file, neg_file])
    label_vectorizer = LabelVectorizer(load_pickled(cluster_labels_file))
    stemmer = MemoizedStemmer()

    pos_provider = TrainingSampleProvider(pos_file, 1, vocab, tfidf, label_vectorizer, stemmer)
    neg_provider = TrainingSampleProvider(neg_file, -1, vocab, tfidf, label_vectorizer, stemmer)

    merged = SampleMerger(pos_provider, neg_provider)

    validation_provider = ValidationSampleProvider(validation_file, None, vocab, tfidf, label_vectorizer, stemmer)

    return merged, validation_provider

def train_xgb(vocab_file, pos_file, neg_file, cluster_labels_file, validation_file, submission_file, gradient_boost_seed, sampler_seed):

    np.random.seed(sampler_seed)

    train, valid = train_setup(vocab_file, pos_file, neg_file, cluster_labels_file, validation_file)

    xgb = GradientBoostingClassifier(
            loss='deviance',
            n_estimators=1000,
            max_depth=5,
            subsample=1.0,
            verbose=1,
            random_state=gradient_boost_seed)

    X, y = samples_to_matrix(train.samples())

    xgb.fit(X, y)

    Xv, _ = samples_to_matrix(valid.samples())

    preds = xgb.predict(Xv.toarray())

    count = 1
    with open(submission_file, 'w', encoding='utf-8') as f:
        f.write('Id,Prediction\n')

        for pred in preds:
            if pred < 0:
                f.write(str(count) + ',' + '-1\n')
            else:
                f.write(str(count) + ',' + '1\n')

            count = count + 1


if __name__ == "__main__":
    pass
