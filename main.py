"""
Thai Word Segmentation and POS Tagging with Deep Learning
"""

import csv
import gc
import glob
import os
import shutil
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
from sklearn.exceptions import UndefinedMetricWarning

import constant
from callback import CustomCallback
from metric import custom_metric
from model import Model
from utils import Corpus, InputBuilder, DottableDict, index_builder

def train(corpus_directory, word_delimiter="|", tag_delimiter="/",
          new_model=True, model_path=None, num_step=60, valid_split=0.1,
          initial_epoch=None, epochs=100, batch_size=32, learning_rate=0.001,
          shuffle=False, es_enable=True, es_min_delta=0.00001, es_patience=5):
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
    char_index = index_builder(constant.CHARACTER_LIST,
                               constant.CHAR_START_INDEX)
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
        initial_epoch = 0
        model = Model(hyper_params).model

    else:
        if not model_path:
            raise Exception("Model path is not defined.")

        if initial_epoch is None:
            raise Exception("Initial epoch is not defined.")

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
              initial_epoch=initial_epoch, epochs=epochs,
              batch_size=batch_size, shuffle=shuffle, callbacks=callbacks)

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
    inb = InputBuilder(test_dataset, char_index, tag_index, model_num_step,
                       y_one_hot=False)
    x_true = inb.x
    y_true = inb.y

    # Load model
    model = load_model(model_path)

    # Predict
    y_pred = model.predict(x_true)
    y_pred = np.argmax(y_pred, axis=2)

    # Calculate score
    scores = custom_metric(y_true, y_pred)

    # Display score
    for metric, score in scores.items():
        print("{0}: {1:.4f}".format(metric, score))

def reevaluate(checkpoint_directory, model_num_step, corpus_directory, csv_path=None,
               word_delimiter="|", tag_delimiter="/"):
    """Reevaluate all checkpoint's model accuracy with custom metrics"""

    # Default csv path
    if not csv_path:
        csv_path = os.path.join(checkpoint_directory, "reevaluate_score.csv")

    # CSV Writer
    file = open(csv_path, "w")
    writer = None

    # Load test dataset
    test_dataset = Corpus(corpus_directory, word_delimiter, tag_delimiter)

    # Create index for character and tag
    char_index = index_builder(constant.CHARACTER_LIST, constant.CHAR_START_INDEX)
    tag_index = index_builder(constant.TAG_LIST, constant.TAG_START_INDEX)

    # Generate input
    inb = InputBuilder(test_dataset, char_index, tag_index, model_num_step,
                       y_one_hot=False)
    x_true = inb.x
    y_true = inb.y

    # Find model in checkpoint directory
    checkpoint_directory = glob.escape(checkpoint_directory)
    model_list = sorted(glob.glob(os.path.join(checkpoint_directory, "*.hdf5")))

    # Process Target
    def predict(model_path, x_true, queue):
        # Load model
        model = load_model(model_path)

        # Predict
        y_pred = model.predict(x_true)
        y_pred = np.argmax(y_pred, axis=2)

        # Put predict result to queue
        queue.put(y_pred)

    # Queue
    queue = Queue()

    # Evaluate score on each model
    for model_path in model_list:
        # Spawn process to load model and predict to prevent memory leak
        process = Process(target=predict, args=(model_path, x_true, queue))
        process.start()

        # Get predict result from queue
        y_pred = queue.get()

        # Join process
        process.join()

        # Calculate score
        scores = custom_metric(y_true, y_pred)

        # Model file name
        model_filename = os.path.basename(model_path)

        # Display score
        print("[Model]", model_filename)

        for metric, score in scores.items():
            print("* {0}: {1:.4f}".format(metric, score))

        print("")

        # Create writer once at first time
        if not writer:
            fields = ["model_filename"] + sorted(scores.keys())
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()

        # Create row data
        row = scores
        row["model_filename"] = model_filename

        # Write row to csv
        writer.writerow(row)
        file.flush()

    # Close file
    writer = None
    file.close()

def summary(model_path):
    """Show model summary"""

    # Load model
    model = load_model(model_path)

    # Show model summary
    print("[Model Summary]")
    model.summary()
    print("")

    # Show model config
    print("[Model Config]")
    pprint(model.get_config())

def show(var):
    """Show variable"""

    result = None

    if var == "char_list":
        result = constant.CHARACTER_LIST
    elif var == "tag_list":
        result = constant.TAG_LIST
    elif var == "char_index":
        result = index_builder(constant.CHARACTER_LIST, constant.CHAR_START_INDEX)
    elif var == "tag_index":
        result = index_builder(constant.TAG_LIST, constant.TAG_START_INDEX)

    # Pretty print
    if result:
        pprint(result)

if __name__ == "__main__":
    # Disable TensorFlow warning
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Disable Sklearn UndefinedMetricWarning
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    # Set random seed for numpy
    np.random.seed(constant.SEED)

    # CLI
    fire.Fire({
        "train": train,
        "run": run,
        "test": test,
        "reevaluate": reevaluate,
        "summary": summary,
        "show": show
    })

    # Garbage collection
    gc.collect()
