
from preprocess import build_vocab, DataSet

def main():
    pos_train = './data/train_pos_full.txt'
    neg_train = './data/train_neg_full.txt'
    vocab = './data/vocab.dat'
    inv_vocab = './data/inv_vocab.dat'

    build_vocab([pos_train, neg_train], DataSet.PAD_WORD, vocab, inv_vocab)
    

if __name__ == '__main__':
    main()
