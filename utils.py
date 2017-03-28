"""
Utilities Function
"""

import glob
import math
import os
import re
import string

import numpy as np
from keras.utils.np_utils import to_categorical

import constant

class Text(object):
    def __init__(self, path, filename, content):
        self.path = path
        self.filename = filename
        self.content = content


class Corpus(object):
    """Corpus Manager"""

    def __init__(self, corpus_directory, word_delimiter=None, tag_delimiter=None):
        # Global variable
        self.corpus_directory = corpus_directory
        self.word_delimiter = word_delimiter
        self.tag_delimiter = tag_delimiter
        self.__corpus = list()

        # Load corpus to memory
        self._load()

    def _preprocessing(self, content):
        """Text preprocessing"""

        # Remove new line
        content = re.sub(r"(\r\n|\r|\n)+", r"", content)

        # Convert one or multiple non-breaking space to space
        content = re.sub(r"(\xa0)+", r"\s", content)

        # Convert multiple spaces to only one space
        content = re.sub(r"\s{2,}", r"\s", content)

        # Trim whitespace from starting and ending of text
        content = content.strip(string.whitespace)

        if self.word_delimiter and self.tag_delimiter:
            # Trim word delimiter from starting and ending of text
            content = content.strip(self.word_delimiter)

            # Convert special characters (word and tag delimiter)
            # in text's content to escape character
            find = "{0}{0}{1}".format(re.escape(self.word_delimiter),
                                      re.escape(self.tag_delimiter))
            replace = "{0}{2}{1}".format(re.escape(self.word_delimiter),
                                         re.escape(self.tag_delimiter),
                                         re.escape(constant.ESCAPE_WORD_DELIMITER))
            content = re.sub(find, replace, content)

            find = "{0}{0}".format(re.escape(self.tag_delimiter))
            replace = "{1}{0}".format(re.escape(self.tag_delimiter),
                                      re.escape(constant.ESCAPE_TAG_DELIMITER))
            content = re.sub(find, replace, content)

        # Replace distinct quotation mark into standard quotation
        content = re.sub(r"\u2018|\u2019", r"\'", content)
        content = re.sub(r"\u201c|\u201d", r"\"", content)

        return content

    def _load(self):
        """Load text to memory"""

        corpus_directory = glob.escape(self.corpus_directory)
        file_list = sorted(glob.glob(os.path.join(corpus_directory, "*.txt")))

        for path in file_list:
            with open(path, "r", encoding="utf8") as text:
                # Read content from text file
                content = text.read()

                # Preprocessing
                content = self._preprocessing(content)

                # Create text instance
                text = Text(path, os.path.basename(path), content)

                # Add text to corpus
                self.__corpus.append(text)

    @property
    def count(self):
        return len(self.__corpus)

    def get_filename(self, index):
        return self.__corpus[index].filename

    def get_token_list(self, index):
        """Get list of (word, tag) pair"""

        if not self.word_delimiter or not self.tag_delimiter:
            return list()

        # Get content by index
        content = self.__corpus[index].content

        # Empty file
        if not content:
            return list()

        # Split each word by word delimiter
        token_list = content.split(self.word_delimiter)

        for idx, token in enumerate(token_list):
            # Empty or Spacebar
            if token == "" or token == constant.SPACEBAR:
                word = constant.SPACEBAR
                tag = constant.SPACEBAR_TAG

            # Word
            else:
                # Split word and tag by tag delimiter
                datum = token.split(self.tag_delimiter)
                word = datum[0]
                tag = datum[-2]

                # Replace escape character to proper character
                word = word.replace(constant.ESCAPE_WORD_DELIMITER, self.word_delimiter)
                tag = tag.replace(constant.ESCAPE_TAG_DELIMITER, self.tag_delimiter)

            # Replace token with word and tag pair
            token_list[idx] = (word, tag)

        return token_list

    def get_char_list(self, index):
        """Get character list"""

        # Get content by index
        content = self.__corpus[index].content

        # Empty file
        if not content:
            return list()

        return list(content)

class InputBuilder(object):
    """Input Builder"""

    def __init__(self, corpus, char_index, tag_index, num_step,
                 text_mode=False, x_3d=False, y_one_hot=True):
        # Global variable
        self.corpus = corpus
        self.char_index = char_index
        self.tag_index = tag_index
        self.num_step = num_step
        self.x_3d = x_3d
        self.y_one_hot = y_one_hot

        if not text_mode:
            self.x = list()
            self.y = list()
            self.generate_x_y()

    def get_encoded_char_list(self, index):
        """Get encoded character list"""

        # Get character list from text
        char_list = self.corpus.get_char_list(index)

        encoded_char_list = [self._encode(self.char_index, char,
                                          default_index=constant.UNKNOW_CHAR_INDEX)
                             for char in char_list]

        # Pad and reshape
        encoded_char_list = self._pad(encoded_char_list, self.num_step)

        if self.x_3d:
            encoded_char_list = encoded_char_list.reshape((-1, self.num_step, 1))
        else:
            encoded_char_list = encoded_char_list.reshape((-1, self.num_step))

        return encoded_char_list

    def generate_x_y(self):
        """Generate input and label for training"""

        for corpus_idx in range(self.corpus.count):
            token_list = self.corpus.get_token_list(corpus_idx)

            for word, tag in token_list:
                # Encode x
                encoded_char_list = [self._encode(self.char_index, char,
                                                  default_index=constant.UNKNOW_CHAR_INDEX)
                                     for char in word]
                self.x.extend(encoded_char_list)

                # Encode y
                self.y.extend([constant.NON_SEGMENT_TAG_INDEX] * (len(word) - 1))
                encoded_tag = self._encode(self.tag_index, tag)
                self.y.append(encoded_tag)

        # Pad and reshape x
        self.x = self._pad(self.x, self.num_step)

        if self.x_3d:
            self.x = self.x.reshape((-1, self.num_step, 1))
        else:
            self.x = self.x.reshape((-1, self.num_step))

        # Pad y
        self.y = self._pad(self.y, self.num_step)

        # Convert y to one-hot vector and reshape y
        if self.y_one_hot:
            self.y = to_categorical(self.y, constant.NUM_TAGS)
            self.y = self.y.reshape((-1, self.num_step, constant.NUM_TAGS))

        else:
            self.y = self.y.reshape((-1, self.num_step))

    def _encode(self, index, key, default_index=-1):
        """Encode to index"""

        # Key does not exist in index
        if key not in index:
            # No Default
            if default_index == -1:
                raise Exception("Unknow tag detected! [{0}]".format(key))

            # Default
            else:
                return default_index

        return index[key]

    def _pad(self, arr, num_step):
        """Pad sequence to full fit within network"""

        size = len(arr)
        pad_size = math.ceil(size / num_step) * num_step
        arr_pad = np.zeros(pad_size)
        arr_pad[:size] = arr

        return arr_pad


class DottableDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self


def index_builder(lst, start_index=1, reverse=False):
    """Build index for encoding"""

    index = dict()

    # Create index dict (reserve zero index for non element in index)
    for idx, key in enumerate(lst, start_index):
        # Duplicate index (multiple key same index)
        if isinstance(key, list):
            for k in key:
                if reverse:
                    index[idx] = k
                else:
                    index[k] = idx

        # Unique index
        else:
            if reverse:
                index[idx] = key
            else:
                index[key] = idx

    return index
