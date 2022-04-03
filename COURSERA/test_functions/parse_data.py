import numpy as np
import csv


def parse_data_from_input(filename):
    with open(filename) as file:
        ### START CODE HERE

        # Use csv.reader, passing in the appropriate delimiter
        # Remember that csv.reader can be iterated and returns one line in each iteration
        csv_reader = csv.reader(file, delimiter=',')

        labels = np.array([])
        images = np.array([])
        for row in csv_reader:
            labels.stack(row[0])
            images.stack(row[1:])

        ### END CODE HERE

        return images, labels