
import numpy as np

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

def write_prediction(filename, pred):
    with open(filename, 'w') as f:
        f.write("Id,Prediction\n")
        for row, col, rec in pred:
            pos_str = "r" + str(row) + "_c" + str(col)
            f.write(pos_str + "," + str(rec) + "\n")

def main():
    gen = parse_input('data/data_train.csv')
    to_predict = parse_input('submissions/sampleSubmission.csv')
    A = fill_matrix(gen, 10000, 1000)

    naive_predictions = predict_naive(A, to_predict)

    write_prediction("submissions/naive_submission.csv", naive_predictions)

if __name__ == "__main__":
    main()
