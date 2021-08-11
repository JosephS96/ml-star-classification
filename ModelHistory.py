from sklearn.metrics import confusion_matrix

class ModelHistory:
    def __init__(self, n_epochs, show_progress=True):
        self.n_epochs = n_epochs

        self.__accuracy = []
        self.__f1_score = []
        self.__loss = []

        self.__show_progress = show_progress

    def get_all_metrics(self):
        return self.__accuracy, self.__f1_score, self.__loss

    def save_all_metrics(self, accuracy, f1_score, loss):
        self.__accuracy.append(accuracy)
        self.__f1_score.append(f1_score)
        self.__loss.append(loss)

        if self.__show_progress:
            self.print_epoch_metrics()

    def print_epoch_metrics(self):
        print(f"Epoch {len(self.__accuracy)}/{self.n_epochs} \n")
        print(
            f"[==============] - loss: {self.__loss[-1]} - accuracy: {self.__accuracy[-1]} - f1 score: {self.__f1_score[-1]}")

    @property
    def accuracy(self):
        return self.accuracy

    @accuracy.setter
    def accuracy(self, value):
        self.__accuracy.append(value)

    @property
    def f1_score(self):
        return self.__f1_score

    @f1_score.setter
    def f1_score(self, value):
        self.__f1_score.append(value)

    @property
    def loss(self):
        return self.__loss

    @loss.setter
    def loss(self, value):
        self.__loss.append(value)

    @property
    def show_progress(self):
        return self.__show_progress

    @show_progress.setter
    def show_progress(self, value):
        self.__show_progress = value
