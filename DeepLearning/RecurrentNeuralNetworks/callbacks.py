from keras.callbacks import Callback
from plots import plot, evaluate_model, save_prediction_examples
import os

class PlotingCallback(Callback):
    """
    Generate and save the plots for the given validation data each period epochs. As Callback
    """
    def __init__(self, path, validation_data, period=1):
        super().__init__()
        self.path = path
        self.validation_data = validation_data
        self.period = period

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.period == 0 and epoch != 0:
            [os.remove(os.path.join(self.path, file)) for file in os.listdir(self.path)]
            used_metrics = [metric for metric in self.model.history.params['metrics'] if 'val' not in metric]
            for label in used_metrics:
                plot(history=self.model.history, path=self.path, label=label)
            evaluate_model(model=self.model, path=self.path, dataset=self.validation_data, split='Validation',
                           target_names=self.validation_data.classes)

class ExtractValidationExamplesCallback(Callback):
    """
    Generate and save the predictions examples for the given validation data each period epochs. As Callback
    """
    def __init__(self, path, validation_data, period=1, predictions_to_save = 10):
        super().__init__()
        self.path = path
        self.validation_data = validation_data
        self.period = period
        self.predictions_to_save = predictions_to_save

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.period == 0 and epoch != 0:
            [os.remove(os.path.join(self.path, file)) for file in os.listdir(self.path)]
            save_prediction_examples(path=self.path, model=self.model, dataset=self.validation_data,
                                     predictions_to_save=self.predictions_to_save)



