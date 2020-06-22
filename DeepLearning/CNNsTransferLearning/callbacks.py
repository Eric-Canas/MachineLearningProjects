from keras.callbacks import Callback
from plots import plot, evaluate_model
import os

class PlotingCallback(Callback):
    """
    Generate and save the plots for the given validation data each period epochs. As Callback
    """
    def __init__(self, path, period=1):
        super().__init__()
        self.path = path
        self.period = period

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.period == 0 and epoch != 0:
            [os.remove(os.path.join(self.path, file)) for file in os.listdir(self.path)]
            used_metrics = [metric for metric in self.model.history.params['metrics'] if 'val' not in metric]
            for label in used_metrics:
                plot(history=self.model.history, path=self.path, label=label)


