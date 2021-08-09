import csv


class DatasetWrapper:
    def __init__(self):
        self.path = "dataset/Stars.csv"
        self.dataset = None

        self.training_data = None
        self.training_labels = None

        self.test_data = None
        self.test_labels = None

        with open(self.path, newline='\n') as csvfile:
            self.dataset = list(csv.reader(csvfile))

    # Reset the data and shuffle it
    def reset(self):
        # Reset current data
        self.training_labels = []
        self.training_data = []
        self.test_data = []
        self.test_labels = []



    def format_data(self):
        return 0

    def get_training_data(self):

        return self.training_data,  self.training_labels

    def get_testing_data(self):

        return self.test_data, self.test_labels

    def get_item(self, index):
        return self.dataset[index]

    def get_len(self):
        return len(self.dataset)
