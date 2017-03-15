import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, \
                            CSVLogger

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
