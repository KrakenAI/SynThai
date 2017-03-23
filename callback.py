
import os
import numpy as np
from metric import custom_metric
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, \
                            TensorBoard, CSVLogger, ProgbarLogger


class CustomMetric(Callback):
    def on_epoch_end(self, epoch, logs):
        # Validation data
        x = self.validation_data[0]
        y_true = self.validation_data[1]

        # Convert 3d one-hot to 2d one-hot
        y_true = np.argmax(y_true, axis=2)

        # Predict
        y_pred = self.model.predict(x)
        y_pred = np.argmax(y_pred, axis=2)

        # Calculate score
        scores = custom_metric(x, y_true, y_pred)

        # Display score
        print(end="\r")
        for metric, score in scores.items():
            print("| {0}: {1:.4f} |".format(metric, score), sep="", end="")

        print("\n")


class Callback(object):
    def __init__(self, params):
        # Callback list
        self.callbacks = list()

        # Callback; Model checkpoint
        if params.checkpoint_enabled:
            cb = ModelCheckpoint(os.path.join(params.cp_dir,
                                              '{epoch:04d}-{val_loss:.4f}.hdf5'),
                                 monitor='val_loss',
                                 save_best_only=params.cp_save_best_only)
            self.callbacks.append(cb)

        # Callback; Early stopping
        if params.es_enabled:
            cb = EarlyStopping(monitor='val_loss', patience=params.es_patience,
                               min_delta=params.es_min_delta,
                               verbose=params.es_verbose)
            self.callbacks.append(cb)

        #TODO: ReduceLROnPlateau

        # Callback; TensorBoard
        if params.tensorboard_enabled:
            cb = TensorBoard(log_dir=params.tensorboard_log_dir,
                             histogram_freq=params.histogram_freq,
                             write_graph=params.write_graph,
                             write_images=params.write_images)
            self.callbacks.append(cb)

        # Callback; CSV log
        if params.csv_log_enabled:
            cb = CSVLogger(filename=params.csv_log_path,
                           separator=params.csv_sep)
            self.callbacks.append(cb)

        # Custom Metric
        cb = CustomMetric()
        self.callbacks.append(cb)
