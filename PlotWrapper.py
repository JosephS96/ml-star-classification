import matplotlib.pyplot as plt


class PlotWrapper:
    def __init__(self, smoothing=0):
        self.smoothing = smoothing

    def plot(self, histories, title, y_label, x_label):

        if self.smoothing != 0:
            self.smooth_values(histories)

        for item in histories:
            plt.plot(item)

        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        #plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def smooth_values(self, histories):
        return histories
