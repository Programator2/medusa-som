#!/usr/bin/env python3
from som import SOM
from sys import argv
import matplotlib.pyplot as plt
from pickle import dump
from pprint import pprint

def main():
    data = set()
    number_input_vectors = 0
    with open(argv[1]) as f:
        for l in f:
            entry = l.split()
            entry.extend([0, 1, 1])
            data.add(tuple(entry))
            number_input_vectors += 1
    data = list(data)
    print('Original input length:', number_input_vectors)
    print('Without duplicates:', len(data))
    s = SOM(data)
    s.train_batch(3)

    with open('pickled_som', 'wb') as f:
        dump(s, f)

    distance_map = s.distance_map()
    plt.pcolor(distance_map, cmap='bone_r')
    plt.axis([0, s.col_sz, 0, s.row_sz])
    plt.show()


if __name__ == '__main__':
    main()
