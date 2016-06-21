
import sys

from preprocess import build_cooc, load_pickled

def main(vocab_file, infiles):
    vocab = load_pickled(vocab_file)

    build_cooc(infiles, vocab, 'cooc.dat')

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
