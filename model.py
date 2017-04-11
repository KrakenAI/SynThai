"""
Keras Model
"""

from keras.models import Sequential
from keras.layers import Embedding, LSTM, TimeDistributed, Dense, Dropout
from keras.layers.wrappers import Bidirectional
from keras.optimizers import RMSprop

import constant

class Model(object):
    def __init__(self, hyper_params):
        # Sequential model
        model = Sequential()

        # Embedding layer
        model.add(Embedding(constant.NUM_CHARS, 10,
                            input_length=hyper_params.num_step))

        for _ in range(3):
            # LSTM layer
            lstm = LSTM(128, return_sequences=True, unroll=True,
                        dropout=0.5, recurrent_dropout=0.5)

            # Bidirectional LSTM
            bi_lstm = Bidirectional(lstm)
            model.add(bi_lstm)

            # LSTM dropout
            model.add(Dropout(0.5))

        # RNN
        model.add(TimeDistributed(Dense(constant.NUM_TAGS, activation="softmax"),
                                  input_shape=(hyper_params.num_step, 128)))

        # Optimizer
        optimizer = RMSprop(hyper_params.learning_rate)

        # Compile
        model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                      metrics=["categorical_accuracy"])

        self.model = model
