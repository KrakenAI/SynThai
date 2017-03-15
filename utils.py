import constant
import os
import glob
import string
import re
import math
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

class Text(object):
    def __init__(self, path, content):
        self.path = path
        self.content = content


class Corpus(object):
    def __init__(self, directory_path, word_delimiter=None, tag_delimiter=None):
        self.directory_path = directory_path
        self.word_delimiter = word_delimiter
        self.tag_delimiter = tag_delimiter
        self.__corpus = list()

        # Load corpus to memory
        self._load()

    def _preprocessing(self, content):
        # Remove new line
        content = re.sub(r"(\r\n|\r|\n)+", r"", content)

        # Convert one or multiple non-breaking space to space
        content = re.sub(r"(\xa0)+", r"\s", content)

        # Convert multiple spaces to one space
        content = re.sub(r"\s{2,}", r"\s", content)

        # Trim whitespace from starting and ending
        content = content.strip(string.whitespace)

        if self.word_delimiter and self.tag_delimiter:
            # Trim word delimiter from starting and ending
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

        # Replace distinct quotation mark into standard
        content = re.sub(r"\u2018|\u2019", r"\'", content)
        content = re.sub(r"\u201c|\u201d", r"\"", content)

        return content

    def _load(self):
        directory_path = glob.escape(self.directory_path)
        path_name = os.path.join(directory_path, "*.txt")
        file_list = sorted(glob.glob(path_name))

        for path in file_list:
            with open(path, "r", encoding="utf8") as text:
                # Read content from text file
                content = text.read()

                # Preprocessing
                content = self._preprocessing(content)

                # Create content instance
                text = Text(path, content)

                # Add text to corpus
                self.__corpus.append(text)

    @property
    def count(self):
        return len(self.__corpus)

    def get_token_list(self, index):
        if not self.word_delimiter or not self.tag_delimiter:
            return list()

        # Grab content by index
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
                #TODO: Assert token length
                datum = token.split(self.tag_delimiter)
                word = datum[0]
                tag = datum[1]

                # Replace escape character to proper character
                word = word.replace(constant.ESCAPE_WORD_DELIMITER, self.word_delimiter)
                tag = tag.replace(constant.ESCAPE_TAG_DELIMITER, self.tag_delimiter)

            # Replace token with word and tag pair
            token_list[idx] = (word, tag)

        return token_list

    def get_char_list(self, index):
        # Grab content by index
        content = self.__corpus[index].content

        # Empty file
        if not content:
            return list()

        return list(content)

class InputBuilder(object):
    def __init__(self, corpus, char_index, tag_index, num_steps,
                 text_mode=False, three_dimension=False):
        # Global Variable
        self.corpus = corpus
        self.char_index = char_index
        self.tag_index = tag_index
        self.num_steps = num_steps
        self.three_dimension = three_dimension

        self.x = list()

        if not text_mode:
            self.y = list()
            self.generate_x_y()

        else:
            self.generate_x()

    def generate_x(self):
        # Generate x from text
        for corpus_idx in range(self.corpus.count):
            char_list = self.corpus.get_char_list(corpus_idx)

            encode_word = [self._encode(self.char_index, char,
                                        default_index=constant.UNKNOW_CHAR_INDEX)
                           for char in char_list]

            self.x.extend(encode_word)

        # Pad and reshape x
        self.x = self._pad(self.x, self.num_steps)

        if self.three_dimension:
            self.x = self.x.reshape((-1, self.num_steps, 1))
        else:
            self.x = self.x.reshape((-1, self.num_steps))

    def generate_x_y(self):
        # Generate x, y from corpus
        for corpus_idx in range(self.corpus.count):
            token_list = self.corpus.get_token_list(corpus_idx)

            for word, tag in token_list:
                # x
                encode_word = [self._encode(self.char_index, char,
                                            default_index=constant.UNKNOW_CHAR_INDEX)
                               for char in word]

                self.x.extend(encode_word)

                # y
                self.y.extend([constant.NON_SEGMENT_TAG_INDEX] * (len(word) - 1))
                encode_tag = self._encode(self.tag_index, tag,
                                          default_index=constant.UNKNOW_TAG_INDEX)
                self.y.append(encode_tag)

        # Pad and reshape x
        self.x = self._pad(self.x, self.num_steps)

        if self.three_dimension:
            self.x = self.x.reshape((-1, self.num_steps, 1))
        else:
            self.x = self.x.reshape((-1, self.num_steps))

        # Pad, convert to one-hot vector, and reshape y
        self.y = self._pad(self.y, self.num_steps)
        self.y = to_categorical(self.y, constant.NUM_TAGS)
        self.y = self.y.reshape((-1, self.num_steps, constant.NUM_TAGS))

    def _encode(self, index, key, default_index=0):
        # Key does not exist in index
        if key not in index:
            return default_index

        return index[key]

    def _pad(self, arr, num_steps):
        # Pad to fit for num_steps dimension reshaping
        size = len(arr)
        pad_size = math.ceil(size / num_steps) * num_steps
        arr_pad = np.zeros(pad_size)
        arr_pad[:size] = arr

        return arr_pad



def index_builder(lst, start_index=1, reverse=False):
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
