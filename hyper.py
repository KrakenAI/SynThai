"""
Hyperopt
"""

import gc
import os
import pickle
import sys
import warnings
from datetime import datetime
from multiprocessing import Process, Queue
from pprint import pprint

# Prevent Keras info message; "Using TensorFlow backend."
STDERR = sys.stderr
sys.stderr = open(os.devnull, "w")
from keras.models import load_model
sys.stderr = STDERR

import fire
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, TimeDistributed, Dense, Dropout
from keras.layers.wrappers import Bidirectional
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split

import constant
from callback import CustomCallback
from utils import Corpus, InputBuilder, DottableDict, index_builder

# Hyper-parameters space
space = {
    'embedding_neuron': hp.choice('embedding_neuron', [2, 5, 10]),
    'lstm': hp.choice('lstm', [
        {
            'layer': 1,
            'neuron': [hp.choice('neuron_11', [32, 64, 96, 128, 256])],
            'dropout': [hp.choice('dropout_11', [0.1, 0.25, 0.5])]
        },
        {
            'layer': 2,
            'neuron': [
                hp.choice('neuron_21', [32, 64, 96, 128, 256]),
                hp.choice('neuron_22', [32, 64, 96, 128, 256])
            ],
            'dropout': [
                hp.choice('dropout_21', [0.1, 0.25, 0.5]),
                hp.choice('dropout_22', [0.1, 0.25, 0.5])
            ]
        },
        {
            'layer': 3,
            'neuron': [
                hp.choice('neuron_31', [32, 64, 96, 128, 256]),
                hp.choice('neuron_32', [32, 64, 96, 128, 256]),
                hp.choice('neuron_33', [32, 64, 96, 128, 256])
            ],
            'dropout': [
                hp.choice('dropout_31', [0.1, 0.25, 0.5]),
                hp.choice('dropout_32', [0.1, 0.25, 0.5]),
                hp.choice('dropout_33', [0.1, 0.25, 0.5])
            ]
        },
        {
            'layer': 4,
            'neuron': [
                hp.choice('neuron_41', [32, 64, 96, 128, 256]),
                hp.choice('neuron_42', [32, 64, 96, 128, 256]),
                hp.choice('neuron_43', [32, 64, 96, 128, 256]),
                hp.choice('neuron_44', [32, 64, 96, 128, 256])
            ],
            'dropout': [
                hp.choice('dropout_41', [0.1, 0.25, 0.5]),
                hp.choice('dropout_42', [0.1, 0.25, 0.5]),
                hp.choice('dropout_43', [0.1, 0.25, 0.5]),
                hp.choice('dropout_44', [0.1, 0.25, 0.5])
            ]
        },
    ]),
    'optimizer': hp.choice('optimizer', ["RMSprop", "Adam"]),
    'batch_size': hp.choice('batch_size', [32, 64, 128])
}

# Global variable
num_step = None
epochs = None
shuffle = None

# Global dataset for hyperopt
x_train = None
y_train = None
x_test = None
y_test = None

def model(params):
    # Queue
    queue = Queue()

    # Initialize checkpoint directory
    directory_name = datetime.today().strftime("%d-%m-%Y-%H-%M-%S")
    checkpoint_directory = os.path.join("checkpoint", directory_name)

    # Process Target
    def train(params, checkpoint_directory, queue):
        # Hyper-parameters
        embedding_neuron = params['embedding_neuron']
        lstm_params = params['lstm']
        lstm_num_layer = lstm_params['layer']
        optimizer = params['optimizer']
        batch_size = params['batch_size']

        # Debug
        print("[Params]", params)

        # Initialize checkpoint directory
        tensorboard_directory = os.path.join(checkpoint_directory, "tensorboard")
        os.makedirs(checkpoint_directory)
        os.makedirs(tensorboard_directory)

        # Sequential model
        model = Sequential()

        # Embedding layer
        model.add(Embedding(constant.NUM_CHARS, embedding_neuron,
                            input_length=num_step))

        for i in range(lstm_num_layer):
            neuron = lstm_params['neuron'][i]
            dropout_rate = lstm_params['dropout'][i]

            # LSTM layer
            lstm = LSTM(neuron, return_sequences=True, unroll=True,
                        dropout=dropout_rate, recurrent_dropout=dropout_rate)

            # Bidirectional LSTM
            bi_lstm = Bidirectional(lstm)
            model.add(bi_lstm)

            # LSTM dropout
            model.add(Dropout(dropout_rate))

        # RNN
        model.add(TimeDistributed(Dense(constant.NUM_TAGS, activation="softmax"),
                                  input_shape=(num_step, lstm_params['neuron'][-1])))

        # Compile
        model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                      metrics=["categorical_accuracy"])

        # Save model architecture to file
        with open(os.path.join(checkpoint_directory, "model.json"), "w") as file:
            file.write(model.to_json())

        # Save model config to file
        with open(os.path.join(checkpoint_directory, "model_config.txt"), "w") as file:
            pprint(model.get_config(), stream=file)

        # Display model summary before train
        model.summary()

        # Callback
        params = DottableDict({
            "es_enable": False,
            "es_min_delta": 0,
            "es_patience": 0
        })
        path = DottableDict({
            "checkpoint": checkpoint_directory,
            "tensorboard": tensorboard_directory,
            "loss_log": os.path.join(checkpoint_directory, "loss.csv"),
            "score_log": os.path.join(checkpoint_directory, "score.csv")
        })
        callbacks = CustomCallback(params, path).callbacks

        # Train
        model.fit(x_train, y_train, validation_data=(x_test, y_test),
                  epochs=epochs, batch_size=batch_size, verbose=2,
                  callbacks=callbacks, shuffle=shuffle)

        # Evaluate
        _, accuracy = model.evaluate(x_test, y_test, verbose=0)

        # Debug
        print("[Validation] categorical_accuracy:", accuracy)
        print("")

        # Put accuracy to queue
        queue.put(accuracy)

    # Spawn process for training model to prevent memory leak
    process = Process(target=train, args=(params, checkpoint_directory, queue))
    process.start()

    # Get accuracy from queue
    accuracy = queue.get()

    return {"loss": -accuracy, "status": STATUS_OK, "params": params,
            "checkpoint_directory": checkpoint_directory}

def hyper(corpus_directory, word_delimiter="|", tag_delimiter="/",
          num_step=60, valid_split=0.1, epochs=5, shuffle=False):
    """Hyperas"""

    # Initialize global variable
    globals()['num_step'] = num_step
    globals()['epochs'] = epochs
    globals()['shuffle'] = shuffle

    # Load train dataset
    train_dataset = Corpus(corpus_directory, word_delimiter, tag_delimiter)

    # Create index for character and tag
    char_index = index_builder(constant.CHARACTER_LIST,
                               constant.CHAR_START_INDEX)
    tag_index = index_builder(constant.TAG_LIST, constant.TAG_START_INDEX)

    # Generate input
    inb = InputBuilder(train_dataset, char_index, tag_index, num_step)
    x_true = inb.x
    y_true = inb.y

    # Split training and validation dataset
    x_train, x_test, y_train, y_test = train_test_split(x_true, y_true,
                                                        test_size=valid_split,
                                                        random_state=constant.SEED)

    # Bind dataset to global variable
    globals()['x_train'] = x_train
    globals()['y_train'] = y_train
    globals()['x_test'] = x_test
    globals()['y_test'] = y_test

    print("[ORIGINAL]", len(x_true), len(y_true))
    print("[SPLIT]", len(x_train), len(y_train), len(x_test), len(y_test))

    # Stop whenever you like (Ctrl+C)
    while True:
        # Initialize Trials
        trials_path = "checkpoint/trials.pickle"

        try:
            trials = pickle.load(open(trials_path, "rb"))
            max_trials = len(trials.trials) + 1

            print("Running trails #{}".format(max_trials))

        except:
            trials = Trials()
            max_trials = 1

            print("Create new trials")

        # Run Hyperopt
        best = fmin(model, space=space, algo=tpe.suggest, max_evals=max_trials,
                    trials=trials)

        # Display best model
        print("[BEST MODEL]")
        print("Checkpoint Directory;", trials.best_trial["result"]["checkpoint_directory"])
        print("Params;", trials.best_trial["result"]["params"])

        # Save Trials
        pickle.dump(trials, open(trials_path, "wb"))

if __name__ == "__main__":
    # Disable TensorFlow warning
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Disable Sklearn UndefinedMetricWarning
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    # Set random seed for numpy
    np.random.seed(constant.SEED)

    # CLI
    fire.Fire(hyper)

    # Garbage collection
    gc.collect()
