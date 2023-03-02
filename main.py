import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product


def sync_network_recall(input, weights, max_iter=1000):
    recall = np.sign(input @ weights)
    for i in range(max_iter):
        prev_output = np.copy(recall)
        recall = np.sign(recall @ weights)
        if np.array_equal(recall, prev_output):
            break
    return recall


def async_network_recall(input, weights, max_iter=1000):
    if len(input.shape) == 1:
        input = [input]
    dim = weights.shape[0]
    recall = np.sign(input @ weights)
    for i in range(max_iter):
        prev_output = np.copy(recall)
        for j in range(dim):
            recall[:, j] = np.sign(recall @ weights[j])
        if np.array_equal(recall, prev_output):
            break
    return recall


def train_hebbian(train_data):
    train_data = np.array(train_data)
    n_samples = train_data.shape[0]
    dim = train_data.shape[1]
    weights = np.zeros((dim, dim))
    for i in range(n_samples):
        weights += np.outer(train_data[i], train_data[i])
    return weights / dim


if __name__ == '__main__':
    # 3.1 Convergence and attractors
    x1 = [-1, -1, 1, -1, 1, -1, -1, 1]
    x2 = [-1, -1, -1, -1, -1, 1, -1, -1]
    x3 = [-1, 1, 1, -1, -1, 1, -1, 1]
    train_data = np.array([x1, x2, x3])

    weights = train_hebbian(train_data)
    output = async_network_recall(train_data, weights)
    print(np.array_equal(train_data, output))

    x1d = [1, -1, 1, -1, 1, -1, -1, 1]
    x2d = [1, 1, -1, -1, -1, 1, -1, -1]
    x3d = [1, 1, 1, -1, 1, 1, -1, 1]
    distorted_data = np.array([x1d, x2d, x3d])

    output = async_network_recall(distorted_data, weights)
    print(np.array_equal(train_data, output))

    attractors = {}
    dim = train_data.shape[1]

    all_inputs = [''.join(x) for x in product('01', repeat=dim)]
    for inputs in all_inputs:
        inputs = np.array([int(x) for x in inputs])
        output = async_network_recall(inputs, weights)
        hash_code = tuple(output[0].tolist())
        if hash_code not in attractors:
            attractors[hash_code] = 1
        else:
            attractors[hash_code] += 1

    print(len(attractors.keys()))

    # 3.2 Sequential update
    p = pd.read_csv('pict.dat', sep=',', header=None)
    p = p.to_numpy().reshape((11, 32, 32))

    # Train and check if patterns are stable
    train_data = p[:3].reshape((3, 32 * 32))
    weights = train_hebbian(train_data)
    output = async_network_recall(train_data, weights)
    print(np.array_equal(output, train_data))

    # Restore degraded pattern
    output = async_network_recall(p[9].reshape((32 * 32)), weights)
    #plt.imshow(output.reshape((32, 32)))
    #plt.show()

    random = np.sign(np.random.random(32 * 32) - 0.5)
    plt.imshow(random.reshape((32, 32)))
    #plt.show()
    output = async_network_recall(random, weights)
    plt.imshow(output.reshape((32, 32)))
    plt.show()


