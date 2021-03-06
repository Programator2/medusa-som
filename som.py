from collections import defaultdict
import math
import operator
import random
from statistics import mean
from pprint import pprint

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


PROCESS_CATEGORY = 0
PATH_CATEGORY = 1
READ_CATEGORY = 2
WRITE_VATEGORY = 3
SEE_CATEGORY = 4


class SOM:
    def __init__(self, data, learning_rate=0.5, ack_threshold=0.5):
        """
        :param data: list of input vectors
        """
        self.data = data

        self.data = [x for x in data if isinstance(x[1], str)]

        self._dimension_check(self.data)

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

        # Stores dictionary mapping from int (id of category) -> list of
        # neurons in that category
        self.neurons_by_cat_id = None

        # Vectors of cluster centers
        self.centers = None

        self.categories = None

    def _dimension_check(self, data: list):
        """
        Checks if all vectors in `data` have the same length as `self.data[0]`
        Raises RuntimeError if not.
        :param data: list of vectors
        """
        size = len(self.data[0])
        if any(map(lambda x: len(x) != size, self.data)):
            raise RuntimeError('Wrong dimension')

    def _random_neuron(self):
        """
        Generate random vector from input data
        """
        neuron = [random.choice(self.data)[i] for i in range(len(self.data[0]))]
        return neuron

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

        slash_num += p1[index:].count('/')
        slash_num += p2[index:].count('/')

        return slash_num

    @staticmethod
    def _simple_distance(x, m):
        """
        :returns: 1 if the values equal, 0 otherwise
        """
        return x == m

    _simple_distance = operator.eq
    _process_distance = _simple_distance

    @staticmethod
    def _distance_category(x: list, m: list, category: int):
        """
        Computes distance of vectors in category
        :param x: input vector
        :param m: input vector
        :param category: category
        """
        funcs = {0: SOM._process_distance,
                 1: SOM.path_distance,
                 2: SOM._simple_distance,
                 3: SOM._simple_distance,
                 4: SOM._simple_distance,
                 }
        return funcs[category](x, m)

    @staticmethod
    def _distance(x: list, m: list):
        """
        Computes distance between vectors x and m.
        """
        distance = int(x[0] == m[0])*3

        distance += SOM.path_distance(x[1], m[1])

        for i in range(2, 5):
            try:
                distance += x[i] == m[i]
            except IndexError:
                breakpoint()

        return distance

    def _best_matching_unit(self, vector):
        return min(
            ((self._distance(vector, neuron), neuron, index) for index, neuron in enumerate(self.neurons)),
            key=lambda x: x[0])

    def _best_matching_units(self):
        """
        :returns: list of best matching units corresponding to each data input
        """
        return [
                self._best_matching_unit(input_vector)[1:] for input_vector in self.data
        ]

    def train(self, iterations):
        """Train the SOM"""
        # TODO Finish
        for i in range(iterations):
            idx = i % (self.n_samples-1)
            self.update(idx)

    def _update_ordinal(self, category, step, best_matching_units):
        """Updates ordinal features. In our case, R, W and S."""
        new_values = [0] * len(self.neurons)


        for p_id, p in enumerate(self.neurons):
            weights, neigh_sum = self._neighbour_weights(step, best_matching_units, p_id)
            allowed_frequency = sum(
                weights[index] for index, input_vector in enumerate(self.data)
                if input_vector[category] == 1
            )/neigh_sum

            # If it's >= 0.5, then it will be allowed, disallowed otherwise
            new_values[p_id] = round(allowed_frequency)

        return new_values

    def _neuron_neighborhood(self) -> dict:
        """
        This function computes a `list` of input vectors for each neuron that
        are closest to that neuron (it is the best matching unit for all
        input vectors in the list).
        """
        input_neighbours = defaultdict(list)
        for p in self.data:
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
        for d in range(1, distance+1):
            for col in range(-d, d+1):
                for row in range(-d, d+1):
                    if not col and not row:
                        # (0, 0) is our neuron
                        continue
                    neighbor_col = neuron_index % self.col_sz + col
                    neighbor_row = neuron_index // self.row_sz + row
                    if not 0 <= neighbor_col < self.col_sz or not 0 <= neighbor_row < self.row_sz:
                        continue
                    index = neuron_index + row * self.col_sz + col
                    ret.append(index)
        return ret

    def _topological_distance(self, neuron_index_a: int, neuron_index_b: int):
        a = neuron_index_a
        b = neuron_index_b

        a = (a % self.col_sz, a // self.row_sz)
        b = (b % self.col_sz, b // self.row_sz)

        return int(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))

    def _neighborhood_set(self, neuron_index: int, distance: int, neuron_neighborhood: dict):
        ret = set(neuron_neighborhood[neuron_index])
        neighborhood = self._topological_neighborhood(neuron_index, distance)
        for index in neighborhood:
            ret.update(neuron_neighborhood[index])
        return ret

    @staticmethod
    def find_median(paths):
        """
        :returns: tuple of (median path: str, error)
        """
        paths = list(paths)
        distance_sum = [0] * len(paths)

        for index, path in enumerate(paths):
            distance_sum[index] = sum(
                SOM.path_distance(path, other_path) for other_path in paths
            )

        # XXX What to do if there are more minims?

        median = min_index(distance_sum)

        return paths[median[0]][PATH_CATEGORY], median[1]

    @staticmethod
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
            try:
                return path + '/' + transformation
            except:
                breakpoint()

    def _update_paths(self, iteration: int):
        """Updates file paths of neurons. Update is not stored immediately, new values
        are returned in a list.
        :param iteration: iteration of the update function
        """

        find_median = self.find_median
        transform = self.transform

        # This is where we store new paths for the neurons, they will be stored
        # in original order
        new_paths = []

        # Counter for neurons with empty neighborhood
        empty_neighborhood = 0

        # Compute neighborhood for each neuron in the *input space*
        neighborhood_for_neuron = self._neuron_neighborhood()

        distance = int((1 - self.col_sz / 3.5) / (self.max_iter - 1) *
                       iteration + self.col_sz / 3.5)
        print('distance is', distance)

        for p_id, p in enumerate(self.neurons):
            # Compute new value for each neuron

            # Get the neighborhood of the neuron (input vectors from the topological neighborhood)
            neighborhood = self._neighborhood_set(p_id, distance, neighborhood_for_neuron)
            neighborhood = list(neighborhood)

            # When there are no neighbours:
            if not neighborhood:
                new_paths.append(p[PATH_CATEGORY])
                empty_neighborhood += 1
                continue

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
                    # print(average, neighbour, transformation)
                    if transformation == '.':
                        breakpoint()
                    counter.update([transformation])

                # Apply the most frequent transformation
                most_frequent = counter.most_common()
                if len(most_frequent) == 1:
                    # There were no ties --- ideal case
                    transformation = most_frequent[0][0]
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
                    new_average = transform(average, t)
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
            # print('appending', average)
            new_paths.append(average)

        # print('_update_paths(): neighborhood was empty', empty_neighborhood, 'times')
        return new_paths

    def _apply_update(self, new_neurons: list):
        """Represents the seconds stage of `SOM.update()`

        :param new_neurons: list containing n lists (where n is number of
        categories of neurons) each containing p values (where p is number of
        neurons in SOM

        """
        distance_sum = 0
        new_neurons = list(map(list, zip(*new_neurons)))
        try:
            # for cat_index, c in enumerate(new_neurons):
            for neuron_index, neuron in enumerate(self.neurons):
                distance_sum += self._distance(neuron, new_neurons[neuron_index])
                neuron = new_neurons[neuron_index]
            print('Summed error:', distance_sum)
        except ValueError as e:
            print(e)
            breakpoint()

    def _neighbour_weights(self, step, best_matching_units, p_id):
        """
        :param p_id: index of neuron
        """
        neigh_sum = 0
        weights = []

        for input_index, input_vector in enumerate(self.data):
            try:
                bmu_id = best_matching_units[input_index][1]
            except TypeError:
                breakpoint()
            distance = self._topological_distance(bmu_id, p_id)
            weight = self.neighbour(distance, step)
            weights.append(weight)
            neigh_sum += weight

        return weights, neigh_sum

    def update(self, step: int):
        """Update consists of two stages:

        1) Compute new value of each neuron for each category.

        2) Apply new values to the neurons.

        :param step: step of the current update iteration
        """
        new_neurons = [0] * len(self.neurons[0])
        # prva kategoria (proces)

        # k is active category
        category = 0

        # F = {}

        new_values = [0] * len(self.neurons)

        # list of best matching unit for each input vector
        best_matching_units = self._best_matching_units()

        for p_id, p in enumerate(self.neurons):
            F = [0] *len(self.processes)
            max_frequency = 0
            max_process_id = 0
            max_process_name = p[category]

            weights, neigh_sum = self._neighbour_weights(step, best_matching_units, p_id)

            for r_id, r in enumerate(self.processes):
                same_category = 0
                for input_index, input_vector in enumerate(self.data):
                    if input_vector[category] == r:
                        same_category += weights[input_index]

                F[r_id] = same_category / neigh_sum

                if F[r_id] > max_frequency:
                    # TODO What if there is more than one maximum?
                    max_frequency = F[r_id]
                    max_process_id = r_id
                    # print('Handling process', p_id, 'New maximum is', r_id)
                    max_process_name = r

            # Compute new value for the neuron

            if max_frequency > (sum(F) - F[max_process_id]):
                new_values[p_id] = max_process_name
            elif random.random() > self.ack_threshold:
                new_values[p_id] = max_process_name
            else:
                new_values[p_id] = p[category]


        new_neurons[category] = new_values

        new_neurons[1] = self._update_paths(step)

        for category in (2, 3, 4):
            new_neurons[category] = self._update_ordinal(category, step, best_matching_units)

        self._apply_update(new_neurons)

    def train_batch(self, num_iteration):
        self.max_iter = num_iteration
        for iteration in range(num_iteration):
            self.update(iteration)

    def distance_map(self):
        matrix = []
        for neuron_index in range(len(self.neurons)):
            if neuron_index % self.col_sz == 0:
                matrix_row = []
                matrix.append(matrix_row)
            center_neuron = self.neurons[neuron_index]
            neighborhood = self._topological_neighborhood(neuron_index, 1)
            distances = [self._distance(center_neuron, self.neurons[i]) for i in neighborhood]
            average = mean(distances)
            matrix_row.append(average)
        return matrix

    def find_clusters(self, n):
        # short names for static methods
        find_median = self.find_median
        transform = self.transform

        # Choose cluster centers randomly from neurons
        centers = random.sample(self.neurons, n)

        improvement = True
        count = 0
        while improvement:
            count += 1
            # print('clustering', count)
            improvement = False
            # Convert centers to tuples so they can be hashed
            centers = [tuple(x) for x in centers]

            # Determine the categories of neurons
            categories = defaultdict(list) # dictionary center -> [neurons]
            for i, neuron in enumerate(self.neurons):
                bmu = min(((
                    self._distance(center, neuron), center, index)
                    for index, center in enumerate(centers)),
                    key=lambda x: x[0])
                categories[bmu[1]].append(neuron)
            self.categories = categories


            # Compute new value for each category
            new_centers = [list(x) for x in centers]

            ### Process ###

            # k is active feature of the vector
            feature = 0

            for center_id, center in enumerate(centers):

                processes = [x[feature] for x in categories.get(center, [])]
                if not processes:
                    continue
                process_counter = BetterCounter(processes)
                most_common_list = process_counter.most_common()
                most_common_process = random.choice(most_common_list)[0]
                if new_centers[center_id][feature] != most_common_process:
                    improvement = True
                new_centers[center_id][feature] = most_common_process

            ### Path ###

            for center_id, center in enumerate(centers):
                # Compute new value for each neuron

                # Get the neighborhood of the neuron (input vectors from the topological neighborhood)
                neighborhood = categories.get(center, [])

                # When there are no neighbours:
                if not neighborhood:
                    # empty_neighborhood += 1
                    continue

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
                        if transformation == '.':
                            breakpoint()
                        counter.update([transformation])

                    # Apply the most frequent transformation
                    most_frequent = counter.most_common()
                    if len(most_frequent) == 1:
                        # There were no ties --- ideal case
                        transformation = most_frequent[0][0]
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
                        new_average = transform(average, t)
                        new_error = SOM.summed_distance(new_average, neighborhood)
                        if new_error < error:
                            average = new_average
                            error = new_error
                            break
                        else:
                            # No transformation was better --- we are finished with
                            # neuron `p`
                            improvement = False

                # This is a new path for the `center_id`-th cluster center
                if new_centers[center_id][1] != average and count < 50:
                    improvement = True
                    # print(center_id, 'improved')
                new_centers[center_id][1] = average

            # Permissions

            centers = new_centers
        self.centers = centers

        # Determine the categories of neurons
        # once again because there are old values from the previous run of the
        # loop
        categories = defaultdict(list) # dictionary center -> [neurons]
        for i, neuron in enumerate(self.neurons):
            bmu = min(((
                self._distance(center, neuron), center, index)
                for index, center in enumerate(centers)),
                key=lambda x: x[0])
            categories[bmu[2]].append(neuron)
        self.neurons_by_cat_id = categories

        return centers

    def neuron_categories(self) -> list:
        """
        :returns: list of len(self.neurons) numbers 0..n-1, where n-1 is
        number of categories. Position in the list corresponds with
        a neuron at that position
        """
        categories = []
        for i, neuron in enumerate(self.neurons):
            bmu_i = min(((
                self._distance(center, neuron), center, index)
                for index, center in enumerate(self.centers)),
                key=lambda x: x[0])[2]
            categories.append(bmu_i)
        return categories

    def output_categories(self, n):
        """
        Creates a file categories.txt, in which all clustered input vectors
        will be printed
        """
        f = open(f'categories-{n}.txt', 'w')
        d = defaultdict(list)
        best_matching_units = self._best_matching_units()
        #breakpoint()
        for index, (neuron, i) in enumerate(best_matching_units):
            d[tuple(neuron)].append(self.data[index])
        for center, neurons in self.categories.items():
            print('Center:', center, file=f)
            for n in neurons:
                pprint(d.get(tuple(n), ['nothing']), stream=f)
            print('-'*80, '\n', '-'*80, file=f)
        f.close()

    def cluster_quality(self):
        """
        Computes the Davies-Bouldin index
        """
        n_clusters = len(self.neurons_by_cat_id)
        total_sum = 0
        for k in self.neurons_by_cat_id:
            max_sum = 0
            max_cat = 0
            for l in self.neurons_by_cat_id :
                if k == l:
                    continue
                sum_value = ((self.centroid_distance(k) +
                              self.centroid_distance(l)) /
                             self.between_clusters_distance(k, l))
                if sum_value > max_sum:
                    max_sum = sum_value
                    max_cat = l
            total_sum += max_sum
        return total_sum / n_clusters

    def centroid_distance(self, category):
        input_vectors = self.neurons_by_cat_id[category]
        center = self.centers[category]
        total_distance = sum(self._distance(x, center) for x in input_vectors)
        return total_distance / len(input_vectors)

    def between_clusters_distance(self, category1, category2):
        c1 = self.centers[category1]
        c2 = self.centers[category2]
        return self._distance(c1, c2)
