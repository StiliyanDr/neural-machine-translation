from collections import Counter
import itertools
from typing import NamedTuple

import nltk

from nmt.utils import default_if_none


class MetaTokens(NamedTuple):
    """
    Bundles up meta tokens needed to represent sentences.
    """
    start: str
    end: str
    unknown: str
    padding: str


class Samples(NamedTuple):
    """
    Groups together the source and target sentences from a corpus of
    samples as well as the dictionaries extracted from them.

    The source and target sentences are separated in two lists
    (attributes `source` and `target`) with corresponding sentences
    having the same positions in the lists. Each sentence is represented
    as a list of tokens (strs) and each of the target sentences is
    surrounded by some start and end meta tokens.

    The tokens in each list of sentences whose number of occurrences
    is greater or equal to a certain threshold are extracted into a
    dictionary which also contains the meta tokens. The keys of the
    dictionary are the tokens and their values are their corresponding
    indexes. The two attributes for these dictionaries are
    `source_dict` and `target_dict` (defaulting to `None` as the dicts
    are optional).
    """
    source: list
    target: list
    source_dict: dict = None
    target_dict: dict = None


_TOKEN_COUNT_THRESHOLD = 3


_FILE_ENCODING = "utf-8"


def prepare(source_path,
            target_path,
            meta_tokens,
            extract_dictionaries=True,
            token_count_threshold=None,
            encoding=None):
    """
    Prepares the given samples (pairs of sentences) for later
    processing.

    Given the paths to the files with samples, this function loads them
    and builds a pickleable object that represents them. This object
    contains the sentence pairs and optionally the dictionaries
    extracted from them.

    :param source_path: a str - the path to the text file with source
    sentences.
    :param target_path: a str - the path to the text file with target
    sentences, that is, the translations of the source sentences.
    :param meta_tokens: an instance of MetaTokens - the meta tokens to
    use when representing the sentences.
    :param extract_dictionaries: a boolean value indicating whether to
    extract dictionaries from the sentences. Defaults to True.
    :param token_count_threshold: an int - the min number of occurrences
    of a token in order for it to be included in a dictionary. If
    omitted or None, defaults to TOKEN_COUNT_THRESHOLD.
    :param encoding: a str - the name of the encoding of the files with
    sentences. If omitted or None, defaults to FILE_ENCODING.
    :returns: an instance of Samples.

    :raises IOError: in case of errors while reading the files.
    """
    source, target = (read_corpus(path, encoding)
                      for path in [source_path, target_path])
    source_dict, target_dict = (
        (extract_dictionary_from(sentences,
                                 meta_tokens,
                                 token_count_threshold)
         for sentences in [source, target])
        if (extract_dictionaries)
        else (None, None)
    )
    target = surround_with_meta_tokens(target, meta_tokens)

    return Samples(
        source,
        target,
        source_dict,
        target_dict,
    )


def read_corpus(file_name, encoding=None):
    """
    Reads a corpus of sentences from a text file. The sentences have to
    be separated by a newline character.

    :param file_name: a str - the path to the file to read.
    :param encoding: a str - the encoding of the file. Defaults to
    FILE_ENCODING.
    :returns: a list of sentences, each sentence being a list of strs -
    the tokens in the sentence.
    """
    encoding = default_if_none(encoding, _FILE_ENCODING)

    with open(file_name, encoding=encoding) as file:
        return [nltk.word_tokenize(line)
                for line in file]


def extract_dictionary_from(corpus,
                            meta_tokens,
                            threshold=None):
    """
    Extracts a dictionary of all the tokens in a corpus of sentences.

    :param corpus: an iterable of sentences, each of which is an
    iterable of tokens (strs).
    :param meta_tokens: an instance of MetaTokens. The tokens in the
    corpus are assumed not to contain the meta tokens.
    :param threshold: an int - the min number of occurrences of a token
    in order for it to be included in the dictionary. If omitted or
    None, defaults to TOKEN_COUNT_THRESHOLD.
    :returns: a dict whose keys are the tokens and whose values are
    unique ints - their indexes.
    """
    threshold = default_if_none(threshold, _TOKEN_COUNT_THRESHOLD)
    word_counts = Counter(word
                          for sentence in corpus
                          for word in sentence)
    assert all(w not in meta_tokens
               for w in word_counts)
    words = (word
             for word, count in word_counts.most_common()
             if (count >= threshold))
    all_words = itertools.chain(words, meta_tokens)

    return {w: i for i, w in enumerate(all_words)}


def surround_with_meta_tokens(sentences, meta_tokens):
    """
    :param sentences: an iterable of sentences, each sentence being an
    iterable of tokens (strs).
    :param meta_tokens: an instance of MetaTokens.
    :returns: a list of sentences, each sentence being a list of the
    same tokens as in the corresponding input sentence but with two
    additional tokens - `meta_tokens.start` as the first one and
    `meta_tokens.end` as the last one.
    """
    return [
        [meta_tokens.start, *s, meta_tokens.end]
        for s in sentences
    ]
