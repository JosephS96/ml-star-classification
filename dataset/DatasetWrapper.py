import csv
import random
import util.constants as constants
from util.star_dataset_enums import StarColor


class DatasetWrapper:
    def __init__(self, train_data_split=80):
        self.path = "dataset/Stars.csv"
        self.dataset = None
        self.column_names = []

        self.train_data_split = train_data_split

        self.training_data = []
        self.training_labels = []

        self.test_data = []
        self.test_labels = []

        self.__read_file()

        self.split_index = self.__get_split_index()

    def __get_split_index(self):
        index = (self.train_data_split * self.get_length()) / 100
        return round(index)

    def __read_file(self):
        with open(self.path, newline='\n') as csvfile:
            data = list(csv.reader(csvfile))

        # Remove column headers
        self.column_names = data.pop(0)

        for item in data:
            item[0] = float(item[0])
            item[1] = float(item[1])
            item[2] = float(item[2])
            item[3] = float(item[3])
            item[4] = self.__get_star_color(item[4])
            item[5] = self.__get_spectral_class(item[5])
            item[6] = int(item[6]) # Label with true class

        random.shuffle(data)
        self.dataset = data

    # Reset the data and shuffle it
    def reset(self):
        # Reset current data
        self.training_labels = []
        self.training_data = []
        self.test_data = []
        self.test_labels = []

        random.shuffle(self.dataset)

    def get_training_data(self, one_hot=False):
        for item in self.dataset[:self.split_index]:
            self.training_data.append(item[:-1])
            if one_hot:
                self.training_labels.append(self.to_one_hot_encoding(item[-1], 6))
            else:
                self.training_labels.append(item[-1])

        return self.training_data,  self.training_labels

    def get_testing_data(self):
        for item in self.dataset[self.split_index:]:
            self.test_data.append(item[:-1])
            self.test_labels.append(item[-1])

        return self.test_data, self.test_labels

    def to_one_hot_encoding(self, value, size):
        one_hot = []
        for i in range(size):
            if i == value:
                one_hot.append(1)
            else:
                one_hot.append(0)

        return one_hot

    def get_item(self, index):
        return self.dataset[index]

    def get_length(self):
        return len(self.dataset)

    def __get_star_color(self, color):
        if color in constants.blue_white:
            return StarColor.BLUE_WHITE.value
        elif color in constants.yellow_white:
            return StarColor.YELLOW_WHITE.value
        elif color in constants.blue:
            return StarColor.BLUE.value
        elif color in constants.orange:
            return StarColor.ORANGE.value
        elif color in constants.white:
            return StarColor.WHITE.value
        elif color in constants.yellowish:
            return StarColor.YELLOWISH.value
        elif color in constants.white_yellow:
            return StarColor.WHITE_YELLOW.value
        elif color in constants.red:
            return StarColor.RED.value
        elif color in constants.orange_red:
            return StarColor.ORANGE_RED.value
        elif color in constants.pale_yellow_orange:
            return StarColor.PALE_YELLOW_ORANGE.value

        print("ERROR: Unable to match star color to value")
        return 0

    def __get_spectral_class(self, star_type):
        return {
            'O': 0,
            'B': 1,
            'A': 2,
            'F': 3,
            'G': 4,
            'K': 5,
            'M': 6,
        }[star_type]