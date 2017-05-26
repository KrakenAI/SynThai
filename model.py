"""
Keras Model
"""

from keras.models import Sequential
from keras.layers import Embedding, LSTM, TimeDistributed, Dense, Dropout
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam

import constant

class Model(object):
    def __init__(self, hyper_params):
        # Sequential model
        model = Sequential()

        # Embedding layer
        model.add(Embedding(constant.NUM_CHARS, 5,
                            input_length=hyper_params.num_step))

        # LSTM Layer #1
        lstm = LSTM(256, return_sequences=True, unroll=True,
                    dropout=0.1, recurrent_dropout=0.1)

        model.add(Bidirectional(lstm))
        model.add(Dropout(0.1))

        # LSTM Layer #2
        lstm = LSTM(256, return_sequences=True, unroll=True,
                    dropout=0.1, recurrent_dropout=0.1)

        model.add(Bidirectional(lstm))
        model.add(Dropout(0.1))

        # LSTM Layer #3
        lstm = LSTM(128, return_sequences=True, unroll=True,
                    dropout=0.25, recurrent_dropout=0.25)

        model.add(Bidirectional(lstm))
        model.add(Dropout(0.25))

        # RNN
        model.add(TimeDistributed(Dense(constant.NUM_TAGS, activation="softmax"),
                                  input_shape=(hyper_params.num_step, 128)))

        # Optimizer
        optimizer = Adam(hyper_params.learning_rate)

        # Compile
        model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                      metrics=["categorical_accuracy"])

        self.model = model
