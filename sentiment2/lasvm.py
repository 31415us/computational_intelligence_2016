
import numpy as np

from collections import defaultdict

class Sample(object):
    def __init__(self, tfidf, lda, glove, label):
        self.tfidf = tfidf
        self.lda = lda
        self.glove = glove
        self.label = label

    def kernel(self, b):
        tfidfdot = np.dot(self.tfidf.todense(), b.tfidf.todense().T)
        ldadot = np.dot(self.lda, b.lda)
        glovedot = np.dot(self.glove, b.glove)

        return tfidfdot + ldadot + glovedot

class KernelCache(object):
    
    def __init__(self):
        self.data = {}
        self.key_map = defaultdict(set)

    def get(self, i, j):
        if i < j:
            return self.data[(i, j)]
        else:
            return self.data[(j, i)]

    def set(self, i, j, val):
        if i < j:
            self.data[(i, j)] = val
            self.key_map[i].add((i,j))
            self.key_map[j].add((i,j))
        else:
            self.data[(j, i)] = val
            self.key_map[i].add((j,i))
            self.key_map[j].add((j,i))

    def delete(self, i):
        for key in self.key_map[i]:
            try:
                del self.data[key]
            except KeyError:
                continue
        del self.key_map[i]

class LASVM(object):

    def __init__(self, C, tau):
        self.C = C
        self.tau = tau

        self.b = 0
        self.delta = 0

        self.support_vecs = {}
        self.alphas = {}
        self.grads = {}
        self.kernel_cache = KernelCache()
        self.A = {}
        self.B = {}

        self.vec_id = 0

    def predict(self, sample):
        res = 0
        for s in self.support_vecs:
            res += self.alphas[s] * self.support_vecs[s].kernel(sample)

        res += self.b

        return res

    def seed_support_vecs(self, samples):
        for sample in samples:
            self.add_support_vector(sample)

    def update(self, samples):
        for sample in samples:
            self.lasvm_process(sample)
            self.lasvm_reprocess()

    def finalize(self):
        while self.delta > self.tau:
            self.lasvm_reprocess()

    def lasvm_process(self, sample):
        k = self.add_support_vector(sample)

        i, j = self.propose_violating()

        if sample.label > 0:
            i = k
        else:
            j = k

        if self.tau_violating(i, j):
            self.update_with(i, j)

    def lasvm_reprocess(self):
        i, j = self.propose_violating()

        if not self.tau_violating(i, j):
            return

        self.update_with(i, j)

        i, j = self.propose_violating()

        to_remove = []
        for s in self.support_vecs:
            if not np.abs(self.alphas[s]) < 0.001:
                continue

            if self.support_vecs[s].label < 0 and self.grads[s] >= self.grads[i]:
                to_remove.append(s)

            if self.support_vecs[s].label > 0 and self.grads[s] <= self.grads[j]:
                to_remove.append(s)

        self.b = (self.grads[i] + self.grads[j]) / 2
        self.delta = self.grads[i] - self.grads[j]

        if len(to_remove) is not 0:
            self.remove_support_vecs(to_remove)


    def remove_support_vecs(self, to_remove):
        for s in to_remove:
            del self.support_vecs[s]
            del self.alphas[s]
            del self.grads[s]
            del self.A[s]
            del self.B[s]
            self.kernel_cache.delete(s)


    def update_with(self, i, j):
        lmbd = self.lambda_factor(i, j)
        self.alphas[i] += lmbd
        self.alphas[j] -= lmbd

        for s in self.support_vecs:
            self.grads[s] -= lmbd * (self.kernel_cache.get(i,s) - self.kernel_cache.get(j, s))


    def add_support_vector(self, sample):
        vec_id = self.vec_id
        self.vec_id += 1

        self.support_vecs[vec_id] = sample
        self.alphas[vec_id] = 0
        self.A[vec_id] = min(0, self.C*sample.label)
        self.B[vec_id] = max(0, self.C*sample.label)

        for i in self.support_vecs:
            other = self.support_vecs[i]

            val = sample.kernel(other)

            self.kernel_cache.set(i, vec_id, val)

        acc = 0
        for s in self.support_vecs:
            acc += self.alphas[s] * self.kernel_cache.get(i, vec_id)

        self.grads[vec_id] = sample.label - acc

        return vec_id

    def propose_violating(self):
        max_val = -np.inf
        index = -1
        for s in self.support_vecs:
            if self.alphas[s] < self.B[s]:
                val = self.grads[s]
                if val > max_val:
                    max_val = val
                    index = s

        i = index

        min_val = np.inf
        index = -1
        for s in self.support_vecs:
            if self.alphas[s] > self.A[s]:
                val = self.grads[s]
                if val < min_val:
                    min_val = val
                    index = s

        j = index

        return i, j

    def index_i_chooser(self, s):
        if self.alphas[s] < self.B[s]:
            return self.grads[s]
        else:
            return -np.inf

    def index_j_chooser(self, s):
        if self.alphas[s] > self.A[s]:
            return self.grads[s]
        else:
            return np.inf

    def tau_violating(self, i, j):
        cond1 = self.alphas[i] < self.B[i]
        cond2 = self.alphas[j] > self.A[j]
        cond3 = self.grads[i] - self.grads[j] > self.tau

        return cond1 and cond2 and cond3

    def lambda_factor(self, i, j):
        term1 = (self.grads[i] - self.grads[j]) / (self.kernel_cache.get(i,i) + self.kernel_cache.get(j,j) - 2 * self.kernel_cache.get(i,j))
        term2 = self.B[i] - self.alphas[i]
        term3 = self.alphas[j] - self.A[j]

        return np.min([term1, term2, term3])

