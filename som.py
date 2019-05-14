from collections import defaultdict
import math
import random

from paths import find_transformation
from utils import min_index, BetterCounter

"""
Input vectors will be stored in list of lists like this:

input = [
    [0, 1, ... , 5],
    ...
    [0, 1, ... , 5],
]

For Medusa it should look like this:
['process name', 'file/path', R, W, S]
"""


class SOM:
    def __init__(self, data, learning_rate=0.5, ack_threshold=0.5):
        """
        :param data: list of input vectors
        """
        self.data = data

        # Kolko je unikatnych procesov?
        self.processes = tuple({x[0] for x in self.data})

        self.learning_rate = learning_rate
        self.max_iter = 0
        self.ack_threshold = ack_threshold

        # SOM will be square, size is number of neurons on one edge
        n_samples = len(data)
        size = math.ceil(math.sqrt(5 * math.sqrt(n_samples)))
        self.col_sz = size  # number of columns
        self.row_sz = size  # number of rows

        self.n_samples = n_samples

        # Create neurons for the map, initialize them with random values
        self.neurons = [self._random_neuron() for x in range(size**2)]

    def _random_neuron(self):
        """
        Generate random vector from input data
        """
        return [
            random.choice(self.data)[0],
            random.choice(self.data)[1],
            random.randint(0, 1),
            random.randint(0, 1),
            random.randint(0, 1),
        ]

    def decay(self, step):
        return self.learning_rate / (1+step/(self.max_iter/2))

    def neighbour(self, distance, step):
        return math.exp(-(distance**2/(2*self.decay(step)**2)))

    @staticmethod
    def summed_distance(path, paths):
        """Returns summed distances from path to each path in paths"""
        return sum(SOM.path_distance(path, other_path) for other_path in paths)

    @staticmethod
    def path_distance(p1, p2):
        """Computes distance between two absolute paths (how many nodes apart they are
        in the filesystem tree)
        """
        index = 0
        len_p1 = len(p1)
        len_p2 = len(p2)
        while p1[index] == p2[index]:
            index += 1
            if index >= len_p1 or index >= len_p2:
                # This means that one path is prefix of another path
                # In that case we don't want to add additional two
                # directory jumps
                slash_num = 0
                break
        else:
            slash_num = 2

        slash_num = p1[index:].count('/')
        slash_num += p2[index:].count('/')

        return slash_num

    @staticmethod
    def _distance(x, m):
        distance = x[0] == m[0]

        distance += SOM.path_distance(x[1], m[1])

        for i in range(2, 5):
            distance += x[i] == m[i]

        return distance

    def _best_matching_unit(self, vector):
        return min(
            ((distance(vector, neuron), neuron, index) for index, neuron in enumerate(self.neurons)),
            key=lambda x: x[0])

    def _best_matching_units(self):
        """
        :returns: list of best matching units corresponding to each data input
        """
        return [
            self._best_matching_unit(input_vector)[1] for input_vector in self.data
        ]

    def train(self, iterations):
        """Train the SOM"""
        # TODO Finish
        for i in range(iterations):
            idx = i % (self.n_samples-1)
            self.update()

    def _update_ordinal(self, category, step):
        """Updates ordinal features. In our case, R, W and S."""
        new_values = [0*len(self.neurons)]

        weights, neigh_sum = self._neighbour_weights(category, step)

        for p_id, p in enumerate(self.neurons):
            allowed_frequency = sum(
                weights[index] for index, input_vector in enumerate(self.data)
                if input_vector[category] == 1
            )/neigh_sum

            # If it's >= 0.5, then it will be allowed, disallowed otherwise
            new_values[p_id] = round(allowed_frequency)

        return new_values

    def _neuron_neighborhood(self) -> list:
        """
        This function computes a `list` of input vectors for each neuron that
        are closest to that neuron (it is the best matching unit for all
        input vectors in the list).
        """
        input_neighbours = defaultdict(list)
        for p in self.processes:
            neuron_index = self._best_matching_unit(p)[2]
            input_neighbours[neuron_index].append(p)
        return input_neighbours

    def _topological_neighborhood(self, neuron_index: int, distance: int):
        """
        :param neuron_index: index of the neuron in question in `self.neurons`
        list
        :returns: `list` of neuron *indexes* that are in the rectangular
        neighborhood of neuron identified by `neuron_index` within `distance`
        """
        ret = []
        for d in range(1, distance):
            for col in range(-d, d+1):
                for row in range(-d, d+1):
                    if not col and not row:
                        # (0, 0) is our neuron
                        continue
                    if (not 0 <= col < self.col_sz) and (not 0 <= row < self.row_sz):
                        continue
                    index = row * self.col_sz + col
                    ret.append(index)
        return ret

    def _neighborhood_set(self, neuron_index: int, distance: int, neuron_neighborhood: dict):
        ret = set(neuron_neighborhood[neuron_index])
        neighborhood = self._topological_neighborhood(neuron_index, distance)
        ret.update(neuron_neighborhood[index] for index in neighborhood)
        return ret

    def _update_paths(self, iteration: int):
        """Updates file paths of neurons. Update is not stored immediately, new values
        are returned in a list.
        :param iteration: iteration of the update function
        """

        def find_median(paths):
            """
            :returns: tuple of (median path: str, error)
            """
            distance_sum = [0*len(paths)]

            for index, path in enumerate(paths):
                distance_sum[index] = sum(
                    SOM.path_distance(path, other_path) for other_path in paths
                )

            # XXX What to do if there are more minims?

            median = min_index(distance_sum)

            return paths[median[0]], median[1]

        def transform(path, transformation):
            """Tranforms `path` using `transformation`"""
            if not transformation:
                return path
            elif transformation == '..':
                index = path.rfind('/')
                if index == 0:
                    return '/'
                elif index > 0:
                    return path[:index]
                else:
                    raise ValueError(f'Invalid path: {path}')
            else:
                return path + '/' + transformation

        # This is where we store new paths for the neurons, they will be stored
        # in original order
        new_paths = []

        # Compute neighborhood for each neuron
        neighborhood_for_neuron = self._neuron_neighborhood()

        for p_id, p in enumerate(self.neurons):
            # Compute new value for each neuron

            # Get the neighborhood of the neuron (input vectors that are similar)
            neighborhood = self._neighborhood_set(p_id, 1, neighborhood_for_neuron)

            # Compute average path from the neighborhood
            average, error = find_median(neighborhood)

            improvement = True
            while improvement:
                counter = BetterCounter()

                # Search how the average path needs to be changed to be the
                # same as each of its neighbour. Count the transformations.
                # There are three transformations:
                # 1) go up (..)
                # 2) go down in the directory structure
                # 3) do nothing (paths are equal)
                for neighbour in neighborhood:
                    transformation = find_transformation(average, neighbour)
                    counter.update(transformation)

                # Apply the most frequent transformation
                most_frequent = counter.most_common()
                if len(most_frequent) == 1:
                    # There were no ties --- ideal case
                    transformation = most_frequent[0]
                    if not transformation:
                        # empty string means the paths are equal, we are
                        # finished --- unlikely to happen
                        break
                    transformations = [transformation]
                elif len(most_frequent) > 1:
                    # There were ties. We are going to try each transformation
                    # and find one that gives smaller error value
                    transformations = [x[0] for x in most_frequent]
                else:
                    raise RuntimeError('Unexpected length of most_frequent')

                for t in transformations:
                    new_average = transform(average, most_frequent[0])
                    new_error = SOM.summed_distance(new_average, neighborhood)
                    # TODO Try all possible transformations and choose the best one
                    # disadvantage - slows down the algorithm
                    if new_error < error:
                        average = new_average
                        error = new_error
                        break
                    else:
                        # No transformation was better --- we are finished with
                        # neuron `p`
                        improvement = False

            # This is a new path for the `p_id`-th neuron
            new_paths.append(average)

        return new_paths

    def _apply_update(self, new_neurons: list):
        """Represents the seconds stage of `SOM.update()`

        :param new_neurons: list containing n lists (where n is number of
        categories of neurons) each containing p values (where p is number of
        neurons in SOM

        """
        for cat_index, c in new_neurons:
            for neuron_index, neuron in self.neurons:
                neuron[cat_index] = c[neuron_index]

    def _neighbour_weights(self, category, step):
        neigh_sum = 0
        weights = []

        for input_index, input_vector in enumerate(self.data):
            bmu = best_matching_units[input_index]
            weight = self.neighbour(bmu[category] - p[category],
                                    step)
            weights.append(weight)
            neigh_sum += weight

        return weights, neigh_sum

    def update(self, step: int):
        """Update consists of two stages:

        1) Compute new value of each neuron for each category.

        2) Apply new values to the neurons.

        :param step: step of the current update iteration
        """
        new_neurons = [0*len(self.neurons[0])]
        # prva kategoria (proces)

        # k is active category
        category = 0

        F = {}

        new_values = [0*len(self.neurons)]

        # list of best matching unit for each input vector
        best_matching_units = self._best_matching_units()

        for p_id, p in enumerate(self.neurons):
            F = [0*len(self.processes)]
            max_frequency = 0
            max_process_id = 0

            weights, neigh_sum = self._neighbour_weights(category, step)

            for r_id, r in enumerate(self.processes):
                same_category = 0
                for input_index, input_vector in enumerate(self.data):
                    if bmu[category] == r:
                        same_category += weights[input_index]

                F[r_id] = same_category / neigh_sum

                if F[r_id] > max_frequency:
                    # TODO What if there is more than one maximum?
                    max_frequency = F[r_id]
                    max_process_id = r_id

            # Compute new value for the neuron

            if max_frequency > (sum(F) - F[max_process_id]):
                new_values[p_id] = max_frequency
            elif random.random() > self.ack_threshold:
                new_values[p_id] = max_frequency
            else:
                new_values[p_id] = p[category]

        new_neurons[category] = new_values

        # TODO Add update of path category
        new_neurons[1] = self._update_paths

        for category in (2, 3, 4):
            new_neurons[category] = self._update_ordinal(category, step)

        self._apply_update(new_neurons)

    def train_batch(self, num_iteration):
        self.max_iter = num_iteration
        for iteration in range(num_iteration):
            self.update(iteration)
