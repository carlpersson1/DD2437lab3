import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product


def network_recall(input, weights, max_iter=1000, e=False):
    energies = []
    recall = np.sign(input @ weights)
    energies.append(energy(weights, input))

    for i in range(max_iter):
        prev_output = np.copy(recall)
        recall = np.sign(recall @ weights)
        if np.array_equal(recall, prev_output) and not e:
            break
        energies.append(energy(weights,recall))
    if e:
        return recall, energies

    return recall


def train_hebbian(train_data):
    train_data = np.array(train_data)
    n_samples = train_data.shape[0]
    dim = train_data.shape[1]
    weights = np.zeros((dim, dim))
    for i in range(n_samples):
        weights += np.outer(train_data[i], train_data[i])
    return weights / dim


def energy(w,x):
    e = 0
    for i in range(len(x)):
        for j in range(len(x)):
            e += w[i][j]*x[i]*x[j]
    return -e


if __name__ == '__main__':
    # 3.1 Convergence and attractors
    x1 = [-1, -1, 1, -1, 1, -1, -1, 1]
    x2 = [-1, -1, -1, -1, -1, 1, -1, -1]
    x3 = [-1, 1, 1, -1, -1, 1, -1, 1]
    train_data = np.array([x1, x2, x3])

    weights = train_hebbian(train_data)
    output = network_recall(train_data, weights)
    print(np.array_equal(train_data, output))

    x1d = [1, -1, 1, -1, 1, -1, -1, 1]
    x2d = [1, 1, -1, -1, -1, 1, -1, -1]
    x3d = [1, 1, 1, -1, 1, 1, -1, 1]
    distorted_data = np.array([x1d, x2d, x3d])

    output = network_recall(distorted_data, weights)
    print(np.array_equal(train_data, output))

    attractors = {}
    dim = train_data.shape[1]

    all_inputs = [''.join(x) for x in product('01', repeat=dim)]
    for inputs in all_inputs:
        inputs = [int(x) for x in inputs]
        output = network_recall(inputs, weights)
        hash = tuple(output)
        if hash not in attractors:
            attractors[hash] = 1
        else:
            attractors[hash] += 1

    print(len(attractors.keys()))

    # 3.2 Sequential update
    p = pd.read_csv('pict.dat', sep=',', header=None)
    p = p.to_numpy().reshape((11, 32, 32))

    # Train and check if patterns are stable
    train_data = p[:3].reshape((3, 32 * 32))
    weights = train_hebbian(train_data)
    output = network_recall(train_data, weights)
    print(np.array_equal(output, train_data))

    # Restore degraded pattern
    output = network_recall(p[9].reshape((32 * 32)), weights)
    #plt.imshow(output.reshape((32, 32)))
    #plt.show()

    random = np.sign(np.random.random(32 * 32) - 0.5)
    #plt.imshow(random.reshape((32, 32)))
    #plt.show()
    #output = network_recall(random, weights)
    #plt.imshow(output.reshape((32, 32)))
    #plt.show()

    # 3.3 Energy
    train_data = p[:3].reshape((3, 32 * 32))
    weights = train_hebbian(train_data)

    print("Energy at attractors:")
    for a in train_data:
        print(energy(weights,a))

    dist = p[9:].reshape((2, 32 * 32))
    print("Energy at distorted patterns:")
    for d in dist:
        print(energy(weights,d))

    p11 = p[10].reshape((32 * 32))

    #output, e = network_recall(p11,weights,e=True, max_iter=5)
    #random = np.sign(np.random.random(32 * 32) - 0.5)
    #output, e = network_recall(random,weights,e=True, max_iter=10)

    #plt.plot(range(len(e)), e)
    #plt.xlabel("Iterations")
    #plt.ylabel("Energy")
    #plt.title("Evolution of the energy at the point of the distorded pattern p11 (sequential update)")
    #plt.show()

    random = np.random.choice([-1, 1], 1024)
    weights = np.random.normal(0, 1, (1024, 1024))
    np.fill_diagonal(weights, 0)
    output, e = network_recall(random,weights,max_iter=10, e=True)
    plt.plot(range(len(e)), e)
    plt.xlabel("Iterations")
    plt.ylabel("Energy")
    plt.title("Evolution of the energy at the point of the distorded pattern p11 (sequential update)")
    plt.show()

    weights = 0.5*(weights+weights.T)
    output, e = network_recall(random,weights,max_iter=10, e=True)
    plt.plot(range(len(e)), e)
    plt.xlabel("Iterations")
    plt.ylabel("Energy")
    plt.title("Evolution of the energy at the point of the distorded pattern p11 (sequential update)")
    plt.show()




