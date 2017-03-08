import constant

from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Dropout
from keras.layers.wrappers import Bidirectional
from keras.regularizers import l2
from keras.optimizers import Adam, RMSprop

class Model(object):
    def __init__(self, params):
        # Sequential Model
        model = Sequential()

        # LSTM Layer
        first_layer = True

        for _ in range(params.lstm_num_layers):
            lstm = LSTM(params.lstm_hidden_units, input_length=params.num_steps,
                        return_sequences=True, unroll=True,
                        dropout_W=params.lstm_w_dropout,
                        dropout_U=params.lstm_u_dropout)

            # LSTM Bi-directional
            if params.bidirectional:
                if first_layer:
                    bi_lstm = Bidirectional(lstm, input_shape=(params.num_steps, 1))
                    first_layer = False

                else:
                    bi_lstm = Bidirectional(lstm)

            model.add(bi_lstm)

            # LSTM Dropout
            model.add(Dropout(params.lstm_dropout))

            dim = params.lstm_hidden_units

        # RNN
        model.add(TimeDistributed(Dense(constant.NUM_TAGS, activation="softmax"),
                                  input_shape=(params.num_steps, dim)))

        # Optimizer
        if params.optimizer == "Adam":
            optimizer = Adam(params.learning_rate)
        elif params.optimizer == "RMSprop":
            optimizer = RMSprop(params.learning_rate)

        # Compile
        model.compile(loss=params.loss, optimizer=optimizer, metrics=params.metrics)

        self.model = model
