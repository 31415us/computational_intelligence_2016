
import sys

from preprocess import build_vocab

def main(stopword_file, infiles):
    build_vocab(infiles, stopword_file, 'vocab.dat', 'inv_vocab.dat', cutoff=5)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
