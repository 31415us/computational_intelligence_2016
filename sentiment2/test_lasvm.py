
import numpy as np

import matplotlib.pyplot as plt

from lasvm import Sample, LASVM

def gaussian_sample_generator():
    mu1 = np.array([1,1])
    mu2 = np.array([-1,-1])
    cov1 = np.eye(2)
    cov2 = np.eye(2)

    while True:
        if np.random.uniform() < 0.5:
            vec = np.random.multivariate_normal(mu1, cov1)
            sample = Sample(vec, 1)
        else:
            vec = np.random.multivariate_normal(mu2, cov2)
            sample = Sample(vec, -1)

        yield sample

def plot_dataset(samples):
    x = []
    y = []
    colors = []

    for sample in samples:
        x.append(sample.vec[0])
        y.append(sample.vec[1])
        if sample.label > 0:
            colors.append('r')
        else:
            colors.append('b')

    plt.scatter(x, y, c=colors, marker='o')
    plt.show()

def test_lasvm():
    seed_set = [next(gaussian_sample_generator()) for i in range(0, 20)]
    training_set = [next(gaussian_sample_generator()) for i in range(0, 10000)]

    lasvm = LASVM(1.0, 0.05)

    lasvm.seed_support_vecs(seed_set)

    lasvm.update(training_set)

    validation_set = [next(gaussian_sample_generator()) for i in range(0, 100)]

    count_correct = 0
    for sample in validation_set:
        prediction = lasvm.predict(sample)

        if prediction * sample.label > 0:
            count_correct += 1

    print(len(lasvm.support_vecs))
    print("accuracy: " + str(count_correct / 100))

if __name__ == "__main__":
    test_lasvm()


