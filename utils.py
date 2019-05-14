import collections
from operator import itemgetter


def min_index(l: list):
    """Find minimum
    :returns: tuple of (index, minimum_value)
    """
    min_index = 0
    min_value = l[0]
    for index, value in l:
        if value < min_value:
            min_value = value
            min_index = index

    return min_index, min_value


class BetterCounter(collections.Counter):
    def most_common(self):
        items = sorted(self.items(), key=itemgetter(1))

        max_number = items[-1][1]
        ret = []
        
        while True:
            item = items.pop()
            if item[1] != max_number:
                break
            ret.append(item)

        return ret
