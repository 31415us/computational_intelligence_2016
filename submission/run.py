
from preprocess import build_vocab, build_cooc, load_pickled

from glove import train_glove

from kmeans_embeddings import train_kmeans

from train import train_xgb

def main():
    pos_file = './data/train_pos.txt'
    neg_file = './data/train_neg.txt'
    validation = './data/test_data.txt'
    stopwords = './data/stopwords.txt'

    vocab_file = 'vocab.dat'
    inv_vocab_file = 'inv_vocab.dat'

    cooc_file = 'cooc.dat'

    embeddings_file = 'embeddings.dat'

    label_file = 'labels.dat'

    submission_file = 'submission.csv'

    glove_seed = 1234
    kmeans_seed = 4321
    xgb_seed = 1337
    sampler_seed = 7331


    build_vocab([pos_file, neg_file], stopwords, vocab_file, inv_vocab_file, cutoff=5)

    vocab = load_pickled(vocab_file)
    inv_vocab = load_pickled(inv_vocab_file)

    build_cooc([pos_file, neg_file], vocab, cooc_file)

    train_glove(cooc_file, embeddings_file, glove_seed)

    train_kmeans(embeddings_file, label_file, kmeans_seed)

    train_xgb(
            vocab_file,
            pos_file,
            neg_file,
            label_file,
            validation,
            submission_file,
            xgb_seed,
            sampler_seed)


if __name__ == "__main__":
    main()
