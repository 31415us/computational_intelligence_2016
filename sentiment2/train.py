
import numpy as np

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
    tfidf = TfIdf(vocab, ['./data/train_pos_small.txt', './data/train_neg_small.txt'])
    lda = LdaLoader('topics.lda')
    label_vectorizer = LabelVectorizer(load_pickled('labels.dat'))
    stemmer = MemoizedStemmer()

    pos_provider = TrainingSampleProvider('./data/train_pos_small.txt', 1, vocab, tfidf, lda, label_vectorizer, stemmer)

    return pos_provider



def train():
    vocab = load_pickled('vocab.dat')
    tfidf = TfIdf(vocab, ['./data/train_pos_small.txt', './data/train_neg_small.txt'])
    lda = LdaLoader('topics.lda', 200)
    label_vectorizer = LabelVectorizer(load_pickled('labels.dat'))
    stemmer = MemoizedStemmer()

    pos_provider = TrainingSampleProvider('./data/train_pos_small.txt', 1, vocab, tfidf, lda, label_vectorizer, stemmer)
    neg_provider = TrainingSampleProvider('./data/train_neg_small.txt', -1, vocab, tfidf, lda, label_vectorizer, stemmer)

    merged = SampleMerger(pos_provider, neg_provider).samples()

    svm = LASVM(1, 0.05)

    seed = [next(merged) for _ in range(0, 20)]

    svm.seed_support_vecs(seed)

    svm.update(merged)

    validation_provider = ValidationSampleProvider('./data/test_data.txt', None, vocab, tfidf, lda, label_vectorizer, stemmer)

    count = 1
    with open('submission.csv', 'w', encoding='utf-8') as f:
        f.write('Id,Prediction\n')
        for sample in validation_provider.samples():
            pred = svm.predict(sample)
            if pred < 0:
                f.write(str(count) + ',' + '-1\n')
            else:
                f.write(str(count) + ',' + '1\n')

            count = count + 1

if __name__ == "__main__":
    train()
