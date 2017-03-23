import constant
import os
import shutil
import numpy as np
from utils import Corpus, InputBuilder, index_builder
from keras.models import load_model

class Parameters():
    def __init__(self):
        # Input
        self.text_directory_path = "./text"

        # Output
        self.output_directory_path = "./output"
        self.word_delimiter = "|"
        self.tag_delimiter = "/"

        # Model
        self.model_path = "./checkpoint/15-03-2017-11-40-02/0061-0.1060.hdf5"
        self.model_num_steps = 60

        # Other
        self.seed = 123456789

def main():
    # Initialize parameters
    params = Parameters()

    # Random seed
    np.random.seed(params.seed)

    # create and empty old output directory
    shutil.rmtree(params.output_directory_path, ignore_errors=True)
    os.makedirs(params.output_directory_path)

    # Load text
    texts = Corpus(params.text_directory_path)

    # Create index for character and tag
    char_index = index_builder(constant.CHARACTER_LIST,
                               start_index=constant.CHAR_START_INDEX)
    tag_index = index_builder(constant.TAG_LIST,
                              start_index=constant.TAG_START_INDEX)

    # Generate input
    inb = InputBuilder(texts, char_index, tag_index, params.model_num_steps,
                       text_mode=True)

    # Load model
    model = load_model(params.model_path)

    for text_idx in range(texts.count):
        # Encode text
        x = texts.get_char_list(text_idx)
        encode_x = inb.get_encode_char_list(text_idx)

        # Batch size
        batch_size = encode_x.shape[0]

        # Predict
        pred = model.predict(encode_x, batch_size=batch_size)

        # Get class
        pred = np.argmax(pred, axis=2)

        # Flatten
        pred = pred.flatten()

        # Result list
        result = list()

        for i, c in enumerate(x):
            tag_index = pred[i]

            # Pad
            if tag_index == constant.PAD_TAG_INDEX:
                continue

            # Append character to result list
            result.append(c)

            # Tag at segmented point
            if tag_index != constant.NON_SEGMENT_TAG_INDEX:
                # Tag name
                tag_name = constant.TAG_LIST[tag_index - constant.TAG_START_INDEX]

                # Append tag to result list
                result.append(params.tag_delimiter)
                result.append(tag_name)
                result.append(params.word_delimiter)

        # Merge to text
        text = "".join(result)

        # Save to text file
        filename = texts.filename(text_idx)
        path = os.path.join(params.output_directory_path, filename)

        with open(path, "w") as f:
            f.write(text)
            f.write("\n")

if __name__ == '__main__':
    main()
