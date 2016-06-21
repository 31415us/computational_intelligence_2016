
import sys

from preprocess import build_vocab

def main(infiles):
    build_vocab(infiles, 'vocab.dat', 'inv_vocab.dat')

if __name__ == '__main__':
    main(sys.argv[1:])
