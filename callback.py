"""
Keras Callback
"""

import csv
import os

import numpy as np
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, \
                            TensorBoard, CSVLogger

from metric import custom_metric

class CalcScore(Callback):
    """Calculate score on custom metric with Keras callback"""

    def __init__(self, filename):
        super(CalcScore, self).__init__()
        self.file = open(filename, "w")
        self.writer = None

    def on_epoch_end(self, epoch, logs=None):
        # Validation data
        x_true = self.validation_data[0]
        y_true = self.validation_data[1]

        # Convert 3D one-hot label to 2D label
        y_true = np.argmax(y_true, axis=2)

        # Predict
        y_pred = self.model.predict(x_true)
        y_pred = np.argmax(y_pred, axis=2)

        # Calculate score
        scores = custom_metric(y_true, y_pred)

        # Display score
        print(end="\r")
        for metric, score in scores.items():
            print("| {0}: {1:.4f} |".format(metric, score), sep="", end="")

        # Save score to file
        if not self.writer:
            fields = ["epoch"] + sorted(scores.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=fields)
            self.writer.writeheader()

        row = scores
        row["epoch"] = epoch
        self.writer.writerow(row)
        self.file.flush()

        print("\n")

    def on_train_end(self, logs=None):
        self.file.close()
        self.writer = None


class CustomCallback(object):
    def __init__(self, params, path):
        # Model path
        model_path = os.path.join(path.checkpoint, "{epoch:04d}-{val_loss:.4f}.hdf5")

        # Callbacks
        self.callbacks = [
            ModelCheckpoint(model_path),
            TensorBoard(path.tensorboard),
            CSVLogger(path.loss_log),
            CalcScore(path.score_log)
        ]

        # Early stopping
        if params.es_enable:
            early_stopping = EarlyStopping(monitor="val_loss",
                                           min_delta=params.es_min_delta,
                                           patience=params.es_patience,
                                           verbose=1)
            self.callbacks.append(early_stopping)
