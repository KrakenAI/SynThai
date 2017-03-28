"""
Thai Word Segmentation and POS Tagging with Deep Learning
"""

import os
import shutil
from datetime import datetime
from pprint import pprint

import fire
import numpy as np
from keras.models import load_model

import constant
from callback import CustomCallback
from metric import custom_metric
from model import Model
from utils import Corpus, InputBuilder, DottableDict, index_builder

def train(corpus_directory, new_model=True, model_path=None, num_step=60,
          word_delimiter="|", tag_delimiter="/", valid_split=0.1,
          epochs=1, batch_size=64, learning_rate=0.001, shuffle=False,
          es_enable=False, es_min_delta=0.0001, es_patience=10):
    """Train model"""

    # Initialize checkpoint directory
    directory_name = datetime.today().strftime("%d-%m-%Y-%H-%M-%S")
    checkpoint_directory = os.path.join("checkpoint", directory_name)
    tensorboard_directory = os.path.join(checkpoint_directory, "tensorboard")

    os.makedirs(checkpoint_directory)
    os.makedirs(tensorboard_directory)

    # Load train dataset
    train_dataset = Corpus(corpus_directory, word_delimiter, tag_delimiter)

    # Create index for character and tag
    char_index = index_builder(constant.CHARACTER_LIST, constant.CHAR_START_INDEX)
    tag_index = index_builder(constant.TAG_LIST, constant.TAG_START_INDEX)

    # Generate input
    inb = InputBuilder(train_dataset, char_index, tag_index, num_step)
    x_true = inb.x
    y_true = inb.y

    # Create new model or load model
    hyper_params = DottableDict({
        "num_step": num_step,
        "learning_rate": learning_rate
    })
    if new_model:
        model = Model(hyper_params).model
    else:
        if not model_path:
            raise Exception("Model path is not defined.")

        model = load_model(model_path)

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
        "es_enable": es_enable,
        "es_min_delta": es_min_delta,
        "es_patience": es_patience
    })
    path = DottableDict({
        "checkpoint": checkpoint_directory,
        "tensorboard": tensorboard_directory,
        "loss_log": os.path.join(checkpoint_directory, "loss.csv"),
        "score_log": os.path.join(checkpoint_directory, "score.csv")
    })
    callbacks = CustomCallback(params, path).callbacks

    # Train model
    model.fit(x_true, y_true, validation_split=valid_split,
              epochs=epochs, batch_size=batch_size,
              shuffle=shuffle, callbacks=callbacks)

def run(model_path, model_num_step, text_directory, output_directory,
        word_delimiter="|", tag_delimiter="/"):
    """Run specific trained model for word segmentation and POS tagging"""

    # Create and empty old output directory
    shutil.rmtree(output_directory, ignore_errors=True)
    os.makedirs(output_directory)

    # Load text
    texts = Corpus(text_directory)

    # Create index for character and tag
    char_index = index_builder(constant.CHARACTER_LIST, constant.CHAR_START_INDEX)
    tag_index = index_builder(constant.TAG_LIST, constant.TAG_START_INDEX)

    # Generate input
    inb = InputBuilder(texts, char_index, tag_index, model_num_step, text_mode=True)

    # Load model
    model = load_model(model_path)

    # Display model summary before run
    model.summary()

    # Run on each text
    for text_idx in range(texts.count):
        # Get character list and their encoded list
        x_true = texts.get_char_list(text_idx)
        encoded_x = inb.get_encoded_char_list(text_idx)

        # Predict
        y_pred = model.predict(encoded_x)
        y_pred = np.argmax(y_pred, axis=2)

        # Flatten to 1D
        y_pred = y_pred.flatten()

        # Result list
        result = list()

        # Process on each character
        for idx, char in enumerate(x_true):
            # Character label
            label = y_pred[idx]

            # Pad label
            if label == constant.PAD_TAG_INDEX:
                continue

            # Append character to result list
            result.append(char)

            # Skip tag for spacebar character
            if char == constant.SPACEBAR:
                continue

            # Tag at segmented point
            if label != constant.NON_SEGMENT_TAG_INDEX:
                # Index offset
                index_without_offset = label - constant.TAG_START_INDEX

                # Tag name
                tag_name = constant.TAG_LIST[index_without_offset]

                # Append delimiter and tag to result list
                result.append(tag_delimiter)
                result.append(tag_name)
                result.append(word_delimiter)

        # Save text string to file
        filename = texts.get_filename(text_idx)
        output_path = os.path.join(output_directory, filename)

        with open(output_path, "w") as file:
            # Merge result list to text string and write to file
            file.write("".join(result))
            file.write("\n")

def test(model_path, model_num_step, corpus_directory,
         word_delimiter="|", tag_delimiter="/"):
    """Test model accuracy with custom metrics"""

    # Load test dataset
    test_dataset = Corpus(corpus_directory, word_delimiter, tag_delimiter)

    # Create index for character and tag
    char_index = index_builder(constant.CHARACTER_LIST, constant.CHAR_START_INDEX)
    tag_index = index_builder(constant.TAG_LIST, constant.TAG_START_INDEX)

    # Generate input
    inb = InputBuilder(test_dataset, char_index, tag_index, model_num_step)
    x_true = inb.x
    y_true = inb.y

    # Convert 3D one-hot label to 2D label
    y_true = np.argmax(y_true, axis=2)

    # Load model
    model = load_model(model_path)

    # Display model summary
    model.summary()

    # Predict
    y_pred = model.predict(x_true)
    y_pred = np.argmax(y_pred, axis=2)

    # Calculate score
    scores = custom_metric(y_true, y_pred)

    # Display score
    for metric, score in scores.items():
        print("{0}: {1:.4f}".format(metric, score))

if __name__ == "__main__":
    # Set random seed for numpy
    np.random.seed(constant.SEED)

    fire.Fire({
        "train": train,
        "run": run,
        "test": test
    })
