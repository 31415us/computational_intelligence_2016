
import numpy as np

import snowballstemmer

import pickle

class MemoizedStemmer(object):

    def __init__(self):
        self.stemmer = snowballstemmer.stemmer('english')
        self.lookup = {}

    def stem(self, word):
        res = self.lookup.get(word)
        
        if res is None:
            stemmed = self.stemmer.stemWord(word)
            self.lookup[word] = stemmed
            res = stemmed

        return res

class VocabularyBuilder(object):

    def __init__(self):
        self.current_id = 0
        self.vocab = {}
        self.inv_vocab = {}

    def add(self, word):
        if self.vocab.get(word) is None:
            self.vocab[word] = self.current_id
            self.inv_vocab[self.current_id] = word

            self.current_id += 1

    def add_file(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                for word in line.split():
                        self.add(word)

def build_vocab(infiles, pad_word, voc_out, inv_voc_out):

    stemmer = MemoizedStemmer()
    vb = VocabularyBuilder()

    vb.add(pad_word)

    for filename in infiles:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                for word in line.split():
                    stemmed = stemmer.stem(word)
                    vb.add(stemmed)

    with open(voc_out, 'wb') as f:
        pickle.dump(vb.vocab, f, protocol=-1)

    with open(inv_voc_out, 'wb') as f:
        pickle.dump(vb.inv_vocab, f, protocol=-1)

def load_vocab(vocab_file, inv_vocab_file):

    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)

    with open(inv_vocab_file, 'rb') as f:
        inv_vocab = pickle.load(f)

    return vocab, inv_vocab

def tweet_to_seq(words, vocab):
    return [vocab[word] for word in words]

def longest_sequence(infiles, vocab):
    max_len = -1
    stemmer = MemoizedStemmer()
    for filename in infiles:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                words = [stemmer.stem(w) for w in line.split()]
                seq_len = len(tweet_to_seq(words, vocab))
                if seq_len > max_len:
                    max_len = seq_len

    return max_len + 2


def pad_seq(seq, target_len, pad_word_id):
    seq_len = len(seq)
    for _ in range(0, target_len - seq_len):
        seq.append(pad_word_id)

def file_line_iter(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            yield line

class NumpyArrayWriter(object):

    def __init__(self, filename):
        self.filename = filename

    def append(self, arr):
        with open(self.filename, 'a', encoding='utf-8') as f:
            for elem in arr:
                f.write(str(elem) + ' ')

            f.write('\n')

class DataSet(object):
    PAD_WORD = '<pad>'

    def __init__(self, pos_file, neg_file, vocab_file, inv_vocab_file, validation_x, validation_y, hold_out_proba=0.1):

        self.vocab, self.inv_vocab = load_vocab(vocab_file, inv_vocab_file)

        self.pad_word_id = self.vocab[DataSet.PAD_WORD]

        self.seq_len = longest_sequence([pos_file, neg_file], self.vocab)

        self.pos_samples = file_line_iter(pos_file)
        self.neg_samples = file_line_iter(neg_file)

        self.validation_x = NumpyArrayWriter(validation_x)
        self.validation_y = NumpyArrayWriter(validation_y)

        self.stemmer = MemoizedStemmer()

        self.hold_out_proba = hold_out_proba

    def batches(self, batch_size):
        x_batch = []
        y_batch = []

        while len(x_batch) < batch_size:
            try:
                x, y = next(self.samples())
            except StopIteration:
                break

            if np.random.uniform() < self.hold_out_proba:
                self.save_to_validation(x, y)
            else:
                x_batch.append(x)
                y_batch.append(y)

        if len(x_batch) == 0:
            raise StopIteration()

        yield np.vstack(x_batch), np.vstack(y_batch)

    def save_to_validation(self, x, y):

        self.validation_x.append(x)

        self.validation_y.append(y)

    def samples(self):
        txt = None
        y = None
        while txt is None:
            if self.pos_samples is None and self.neg_samples is None:
                raise StopIteration()
            elif self.pos_samples is None:
                try:
                    txt = next(self.neg_samples)
                    y = np.array([1, 0])
                except StopIteration:
                    self.neg_samples = None
                    raise StopIteration()
            elif self.neg_samples is None:
                try:
                    txt = next(self.pos_samples)
                    y = np.array([0, 1])
                except StopIteration:
                    self.pos_samples = None
                    raise StopIteration()
            else:
                if np.random.uniform() < 0.5:
                    try:
                        txt = next(self.pos_samples)
                        y = np.array([0, 1])
                    except StopIteration:
                        self.pos_samples = None
                else:
                    try:
                        txt = next(self.neg_samples)
                        y = np.array([1, 0])
                    except StopIteration:
                        self.neg_samples = None

        words = [self.stemmer.stem(word) for word in txt.split()]
        seq = tweet_to_seq(words, self.vocab)
        pad_seq(seq, self.seq_len, self.pad_word_id)

        x = np.array(seq)

        yield x, y
        
def read_validation(xval_file, yval_file):
    x = []
    with open(xval_file, 'r', encoding='utf-8') as f:
        for line in f:
            x.append(np.array([int(w) for w in line.split()]))

    y = []
    with open(yval_file, 'r', encoding='utf-8') as f:
        for line in f:
            y.append(np.array([int(w) for w in line.split()]))

    return np.vstack(x), np.vstack(y)

class ValidationSetReader(object):

    def __init__(self, xval, yval):
        self.xs = file_line_iter(xval)
        self.ys = file_line_iter(yval)

    def batches(self, batch_size):
        x_batch = []
        y_batch = []

        while len(x_batch) < batch_size:
            try:
                x, y = next(self.next_sample())
            except StopIteration:
                break

            x_batch.append(x)
            y_batch.append(y)

        if len(x_batch) == 0:
            raise StopIteration

        yield np.vstack(x_batch), np.vstack(y_batch)

    def next_sample(self):
        try:
            xtxt = next(self.xs)
            ytxt = next(self.ys)
        except StopIteration:
            raise StopIteration

        x = np.array([int(w) for w in xtxt.split()])
        y = np.array([int(w) for w in ytxt.split()])

        yield x, y

class PredictionSet(object):

    def __init__(self, to_be_predicted, vocab, seq_len, pad_word):
        self.to_pred = file_line_iter(to_be_predicted)
        self.stemmer = MemoizedStemmer()
        self.vocab = vocab
        self.seq_len = seq_len
        self.pad_word_id = self.vocab[pad_word]

    def batches(self, batch_size):
        x_batch = []

        while len(x_batch) < batch_size:
            try:
                x = next(self.next_tweet())
            except StopIteration:
                break

            x_batch.append(x)

        if len(x_batch) == 0:
            raise StopIteration()

        try:
            np.vstack(x_batch)
        except Exception:
            for x in x_batch:
                print(x.shape)
            print()

        yield np.vstack(x_batch)

    def next_tweet(self):
        try:
            txt = next(self.to_pred)
        except StopIteration:
            raise StopIteration()

        splitted = txt.split(',')
        tweet = splitted[1]

        tweet_stemmed = [self.stemmer.stem(w) for w in tweet.split()]
        id_seq = [self.vocab.get(w) for w in tweet_stemmed if self.vocab.get(w) is not None]

        pad_seq(id_seq, self.seq_len, self.pad_word_id)

        yield np.array(id_seq)
