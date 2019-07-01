#!/usr/bin/env python3
from som import SOM
from sys import argv
import matplotlib.pyplot as plt
from pickle import dump, load
from pprint import pprint
from matplotlib.markers import MarkerStyle

def find_clusters(distance_map, s, n):
    min_clustering = 9999
    centers = s.centers
    categories = s.categories
    # Clustering algorithm is sensitive to initialization
    # Repeat it many times and select the best clustering
    for i in range(300):
        centers = s.find_clusters(n)
        cluster_quality = s.cluster_quality()
        if cluster_quality < min_clustering:
            min_clustering = cluster_quality
            centers = s.centers
            categories = s.categories

    s.centers = centers
    s.categories = categories

    categories = s.neuron_categories()

    plt.figure(n)
    plt.pcolor(distance_map, cmap='bone_r')
    plt.axis([0, s.col_sz, 0, s.row_sz])
    markers = tuple(MarkerStyle.markers)[2:]
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    for neuron_index, neuron in enumerate(s.neurons):
        # print(markers[categories[neuron_index]])
        pos = (neuron_index % s.col_sz, neuron_index // s.row_sz)
        plt.plot(pos[0]+.5, pos[1]+.5, markers[categories[neuron_index]], markerfacecolor='None',
                markeredgecolor=colors[categories[neuron_index]],
                markersize=12, markeredgewidth=2)
    plt.savefig(f'{n}.eps', transparent=True)
    plt.savefig(f'{n}.png', transparent=True)
    s.output_categories(n)
    # plt.show()
    print(f'Clustering {n} finished with {min_clustering}')
    return n, min_clustering

def main():
    # data = set()
    # number_input_vectors = 0
    # with open(argv[1]) as f:
    #     for l in f:
    #         entry = l.split()
    #         entry.extend([0, 1, 1])
    #         data.add(tuple(entry))
    #         number_input_vectors += 1
    # data = list(data)
    # print('Original input length:', number_input_vectors)
    # print('Without duplicates:', len(data))
    # s = SOM(data)
    # s.train_batch(20)

    # with open('pickled_som', 'wb') as f:
    #     dump(s, f)
    with open('pickled_som', 'rb') as f:
         s = load(f)
    s.categories = None

    distance_map = s.distance_map()

    plt.figure(0)
    plt.pcolor(distance_map, cmap='bone_r')
    plt.axis([0, s.col_sz, 0, s.row_sz])
    plt.savefig('umatrix.eps', transparent=True)

    # pprint(s.neurons)

    results = []

    N_CLUSTERS_START = 2
    N_CLUSTERS_END = 7

    for n in range (N_CLUSTERS_START, N_CLUSTERS_END + 1):
        results.append(find_clusters(distance_map, s, n))

    print(results)


if __name__ == '__main__':
    main()
