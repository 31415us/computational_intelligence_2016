
import numpy as np

import scipy.linalg as linalg

def parse_row_col(s):
    row_str, col_str = s.split('_')
    row = int(row_str[1:])
    col = int(col_str[1:])
    return (row, col)

def parse_rec(s):
    pos_str, rec_str = s.split(',')
    row, col = parse_row_col(pos_str)
    rec = int(rec_str)

    return (row, col, rec)

def parse_input(filename):
    with open(filename, 'r') as f:
        f.readline() # drop the first line
        for line in f:
            yield parse_rec(line)

def fill_matrix(gen, num_row, num_col):
    res = np.zeros((num_row, num_col))

    for row, col, rec in gen:
        res[row - 1, col - 1] = rec

    return res

def predict_naive(A, pred_gen):

    B = naive_completion(A)

    for row, col, _ in pred_gen:
        yield (row, col, B[row - 1, col - 1])

def naive_completion(A):

    means = np.average(A, axis=0, weights=A.astype(bool))

    row, col = np.shape(A)

    B = np.zeros((row, col))

    for i in range(0, row):
        for j in range(0, col):
            if A[i,j] == 0:
                B[i,j] = means[j]
            else:
                B[i,j] = A[i,j]

    return B

def svd_based_completion(A, spectrum_cutoff=0.75):

    B = naive_completion(A)

    U, s, Vh = linalg.svd(B, full_matrices=False, compute_uv=True, overwrite_a=True, check_finite=False)

    cutoff_sum = spectrum_cutoff * np.sum(s)

    s_shape = np.shape(s)
    new_s = np.zeros(s_shape)

    acc = 0
    for i in range(0, s_shape[0]):
        acc = acc + s[i]
        if acc >= cutoff_sum:
            break
        else:
            new_s[i] = s[i]

    return np.dot(U, np.dot(np.diag(new_s), Vh))

def predict_svd(A, pred_gen):

    B = svd_based_completion(A)

    for row, col, _ in pred_gen:
        yield (row, col, B[row - 1, col - 1])


def write_prediction(filename, pred):
    with open(filename, 'w') as f:
        f.write("Id,Prediction\n")
        for row, col, rec in pred:
            pos_str = "r" + str(row) + "_c" + str(col)
            f.write(pos_str + "," + str(rec) + "\n")

def create_test_matrix(nrow, ncol, density=0.1):
    # create a random 5 star rating matrix
    full = np.ndarray.astype(5 * np.reshape(np.random.uniform(0, 1, nrow * ncol), (nrow, ncol)) + 1, dtype=int)

    reduced = np.zeros((nrow, ncol))

    for i in range(0, nrow):
        for j in range(0, ncol):
            if np.random.uniform(0, 1) < density:
                reduced[i,j] = full[i,j]

    return full, reduced

def approximation_error(predicted, actual):

    diff = predicted - actual
    
    return np.sqrt(np.sum(diff * diff) / np.size(diff))

def test_method(fun):
    acc = 0
    for i in range(0, 100):
        full, reduced = create_test_matrix(1000, 100)
        pred = fun(reduced)
        acc = acc + approximation_error(pred, full)

    return acc / 100

def main():
    gen = parse_input('data/data_train.csv')
    to_predict = parse_input('submissions/sampleSubmission.csv')
    A = fill_matrix(gen, 10000, 1000)

    #naive_predictions = predict_naive(A, to_predict)

    #write_prediction("submissions/naive_submission.csv", naive_predictions)

    #svd_predictions = predict_svd(A, to_predict)

    #write_prediction("submissions/svd_submission.csv", svd_predictions)

    print(test_method(svd_based_completion))

if __name__ == "__main__":
    main()
