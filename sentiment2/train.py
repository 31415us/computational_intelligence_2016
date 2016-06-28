
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier

from scipy import sparse

from tfidf import TfIdf
from gensim_lda import LdaLoader
from label_vectorizer import LabelVectorizer
from lasvm import LASVM, Sample

from preprocess import MemoizedStemmer, load_pickled

class TrainingSampleProvider(object):

    def __init__(self, filename, label, vocab, tfidf, lda, label_vectorizer, stemmer):
        self.filename = filename
        self.vocab = vocab
        self.tfidf = tfidf
        self.lda = lda
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
                lda_vec = self.lda.topic_vector(filtered)
                label_vec = self.label_vectorizer.vectorize(filtered)

                yield Sample(tfidf_vec, lda_vec, label_vec, self.label)

class ValidationSampleProvider(TrainingSampleProvider):
    def __init__(self, filename, label, vocab, tfidf, lda, label_vectorizer, stemmer):
        super().__init__(filename, label, vocab, tfidf, lda, label_vectorizer, stemmer)

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

def test_provider():
    vocab = load_pickled('vocab.dat')
    tfidf = TfIdf(vocab, ['./data/train_pos.txt', './data/train_neg.txt'])
    lda = LdaLoader('topics.lda', 200)
    label_vectorizer = LabelVectorizer(load_pickled('labels.dat'))
    stemmer = MemoizedStemmer()

    pos_provider = TrainingSampleProvider('./data/train_pos.txt', 1, vocab, tfidf, lda, label_vectorizer, stemmer)

    return pos_provider

def samples_to_matrix(samples):
    X_list = []
    y_list = []

    for sample in samples:
        feature_vec = sparse.hstack([sample.tfidf, sample.lda, sample.glove])
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


def train_setup():
    vocab = load_pickled('vocab.dat')
    tfidf = TfIdf(vocab, ['./data/train_pos.txt', './data/train_neg.txt'])
    lda = LdaLoader('topics.lda', 200)
    label_vectorizer = LabelVectorizer(load_pickled('labels.dat'))
    stemmer = MemoizedStemmer()

    pos_provider = TrainingSampleProvider('./data/train_pos.txt', 1, vocab, tfidf, lda, label_vectorizer, stemmer)
    neg_provider = TrainingSampleProvider('./data/train_neg.txt', -1, vocab, tfidf, lda, label_vectorizer, stemmer)

    merged = SampleMerger(pos_provider, neg_provider)

    validation_provider = ValidationSampleProvider('./data/test_data.txt', None, vocab, tfidf, lda, label_vectorizer, stemmer)

    return merged, validation_provider

def train_xgb():
    train, valid = train_setup()

    xgb = GradientBoostingClassifier(
            loss='deviance',
            n_estimators=10000,
            max_depth=5,
            subsample=1.0,
            verbose=1)

    X, y = samples_to_matrix(train.samples())

    xgb.fit(X, y)

    Xv, _ = samples_to_matrix(valid.samples())

    preds = xgb.predict(Xv)

    count = 1
    with open('submission.csv', 'w', encoding='utf-8') as f:
        f.write('Id,Prediction\n')

        for pred in preds:
            if pred < 0:
                f.write(str(count) + ',' + '-1\n')
            else:
                f.write(str(count) + ',' + '1\n')

            count = count + 1


def train_sgd():
    training, validation = train_setup()

    sgd = SGDClassifier(
            loss='hinge',
            penalty='l2',
            fit_intercept=True,
            n_iter=1,
            shuffle=False,
            n_jobs=-1)

    batch_size = 128

    classes = np.array([1, -1])

    for sample_batch in batch(training.samples(), batch_size):
        X, y = samples_to_matrix(sample_batch)

        sgd.partial_fit(X, y, classes)

    count = 1
    with open('submission.csv', 'w', encoding='utf-8') as f:
        f.write('Id,Prediction\n')

        for validation_batch in batch(validation.samples(), batch_size):

            X, _ = samples_to_matrix(validation_batch)

            preds = sgd.predict(X)

            for pred in preds:
                if pred < 0:
                    f.write(str(count) + ',' + '-1\n')
                else:
                    f.write(str(count) + ',' + '1\n')

                count = count + 1


def train_lasvm():

    training, validation = train_setup()

    training_samples = training.samples()

    svm = LASVM(1, 0.05)

    seed = [next(training_samples) for _ in range(0, 20)]

    svm.seed_support_vecs(seed)

    svm.update(training_samples)

    count = 1
    with open('submission.csv', 'w', encoding='utf-8') as f:
        f.write('Id,Prediction\n')
        for sample in validation.samples():
            pred = svm.predict(sample)
            if pred < 0:
                f.write(str(count) + ',' + '-1\n')
            else:
                f.write(str(count) + ',' + '1\n')

            count = count + 1

if __name__ == "__main__":
    train_xgb()
