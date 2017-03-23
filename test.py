import constant
import os
import shutil
import numpy as np
from metric import custom_metric
from utils import Corpus, InputBuilder, index_builder
from keras.models import load_model

class Parameters():
    def __init__(self):
        # Dataset
        self.test_directory_path = "./corpus/test"
        self.word_delimiter = "|"
        self.tag_delimiter = "/"

        # Model
        self.model_path = "./checkpoint/15-03-2017-11-40-02/0199-0.1019.hdf5"
        self.model_num_steps = 60

        # Other
        self.seed = 123456789

def main():
    # Initialize parameters
    params = Parameters()

    # Random seed
    np.random.seed(params.seed)

    # Load train dataset
    test_dataset = Corpus(params.test_directory_path,
                          params.word_delimiter,
                          params.tag_delimiter)

    # Create index for character and tag
    char_index = index_builder(constant.CHARACTER_LIST,
                               start_index=constant.CHAR_START_INDEX)
    tag_index = index_builder(constant.TAG_LIST,
                              start_index=constant.TAG_START_INDEX)

    # Generate input
    inb = InputBuilder(test_dataset, char_index, tag_index, params.model_num_steps)
    x = inb.x
    y_true = inb.y

    # Load model
    model = load_model(params.model_path)

    # Convert 3d one-hot to 2d one-hot
    y_true = np.argmax(y_true, axis=2)

    # Predict
    y_pred = model.predict(x)
    y_pred = np.argmax(y_pred, axis=2)

    # Calculate score
    scores = custom_metric(x, y_true, y_pred)

    # Display score
    for metric, score in scores.items():
        print("{0}: {1:.4f}".format(metric, score))

if __name__ == '__main__':
    main()
