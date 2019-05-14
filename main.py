#!/usr/bin/env python3
from som import SOM
from sys import argv

def main():
    data = []
    with open(argv[1]) as f:
        for l in f:
            entry = l.split()
            entry.extend([0, 1, 1])
            data.append(entry)
    print(data)


if __name__ == '__main__':
    main()
