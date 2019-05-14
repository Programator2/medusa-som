from som import SOM
from sys import argv

def main():
    data = []
    with open(argv[1]) as f:
        for l in f:
            data.append(l.split().extend([0, 1, 1]))
    print(data)
