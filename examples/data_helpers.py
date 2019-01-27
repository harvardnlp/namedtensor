import re
import sys
import itertools
import numpy as np
from collections import Counter

"""
Adapted from https://github.com/dennybritz/cnn-text-classification-tf
"""


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    x_text, y = load_sentences_and_labels()
    x_text = [s.split(" ") for s in x_text]
    y = [[0, 1] if label == 1 else [1, 0] for label in y]
    return [x_text, y]


def load_sentences_and_labels():
    if sys.version_info.major == 3:
        positive_examples = list(
            open("./data/rt-polarity.pos", encoding="ISO-8859-1").readlines()
        )
        negative_examples = list(
            open("./data/rt-polarity.neg", encoding="ISO-8859-1").readlines()
        )
    else:
        positive_examples = list(open("./data/rt-polarity.pos").readlines())
        negative_examples = list(open("./data/rt-polarity.neg").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = [s.strip() for s in negative_examples]
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    y = positive_labels + negative_labels
    return x_text, y


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array(
        [[vocabulary[word] for word in sentence] for sentence in sentences]
    )
    y = np.array(labels)
    y = y.argmax(axis=1)
    return [x, y]


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]
