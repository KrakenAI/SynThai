import constant
import os
import pprint
import numpy as np
from datetime import datetime
from model import Model
from callback import Callback
from utils import Corpus, InputBuilder, index_builder


class Parameters():
    def __init__(self):
        # Dataset
        self.train_directory_path = "./corpus/train"
        self.test_directory_path = "./corpus/test"
        self.valid_split = 0.1
        self.word_delimiter = "|"
        self.tag_delimiter = "/"

        # Embedding Layer
        self.embedding_hidden_units = 10

        # Network
        self.num_steps = 60
        self.lstm_num_layers = 3
        self.lstm_hidden_units = 128
        self.bidirectional = True

        # Dropout
        self.lstm_dropout = 0.5
        self.lstm_w_dropout = 0.5
        self.lstm_u_dropout = 0.5

        # Optimizer
        self.optimizer = "RMSprop"
        self.learning_rate = 0.001
        self.loss = "categorical_crossentropy"

        # Training
        self.batch_size = 64
        self.n_epoch = 200
        self.shuffle = False

        # Early stopping
        self.es_enabled = False
        self.es_patience = 10
        self.es_min_delta = 0.0001
        self.es_verbose = 1

        # Current time
        now = datetime.today()
        directory_name = now.strftime("%d-%m-%Y-%H-%M-%S")

        # Checkpoint
        self.checkpoint_enabled = True
        self.cp_dir = "./checkpoint/%s" % (directory_name)
        self.cp_save_best_only = False

        # TensorBoard
        self.tensorboard_enabled = True
        self.tensorboard_log_dir = "%s/tensorboard" % (self.cp_dir)
        self.histogram_freq = 0
        self.write_graph = False
        self.write_images = False

        # CSV log
        self.csv_log_enabled = True
        self.csv_log_path = "%s/log.csv" % (self.cp_dir)
        self.csv_sep = ","

        # Debug
        self.metrics = ["categorical_accuracy"]
        self.training_verbose = 1

        # Other
        self.seed = 123456789


def main():
    # Initialize parameters
    params = Parameters()

    # Random seed
    np.random.seed(params.seed)

    # Create directory
    os.makedirs(params.cp_dir)
    os.makedirs(params.tensorboard_log_dir)

    # Load train dataset
    train_dataset = Corpus(params.train_directory_path,
                           params.word_delimiter,
                           params.tag_delimiter)

    # Create index for character and tag
    char_index = index_builder(constant.CHARACTER_LIST,
                               start_index=constant.CHAR_START_INDEX)
    tag_index = index_builder(constant.TAG_LIST,
                              start_index=constant.TAG_START_INDEX)

    # Generate input
    inb = InputBuilder(train_dataset, char_index, tag_index, params.num_steps)

    # Create model
    model = Model(params).model
    callbacks = Callback(params).callbacks

    # Show model summary and config
    pprint.pprint(model.summary())
    pprint.pprint(model.get_config())

    model.fit(inb.x, inb.y, batch_size=params.batch_size,
              validation_split=params.valid_split, epochs=params.n_epoch,
              shuffle=params.shuffle, callbacks=callbacks,
              verbose=params.training_verbose)


if __name__ == "__main__":
    main()
