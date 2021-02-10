import copy

import torch

from nmt.models import utils


class SequenceEncoder(torch.nn.Module):
    """
    A neural network module that can be used as the base for both an
    encoder and a decoder in a sequence-to-sequence architecture.

    This module consists of two layers - an embedding followed by a RNN
    (LSTM).
    """
    def __init__(self,
                 words,
                 meta_tokens,
                 embedding_size,
                 hidden_size,
                 is_bidirectional=False,
                 num_layers=1,
                 dropout=0.):
        """
        :param words: a mapping with words (strs) for keys and indexes
        (ints) for values - the dictionary of the language being encoded.
        :param meta_tokens: the MetaTokens instance used when extracting
        the dictionary.
        :param embedding_size: a positive integer - the size of word
        embeddings.
        :param hidden_size: a positive integer - the size of RNN cells.
        :param is_bidirectional: a boolean value indicating whether the
        RNN should be bidirectional. Defaults to `False`.
        :param num_layers: a positive integer - the number of RNN layers
        (in a single direction). Defaults to 1.
        :param dropout: a float between 0 and 1 - if non-zero,
        introduces a Dropout layer on the outputs of each RNN layer
        except the last one, with dropout probability equal to `dropout`.
        Defaults to 0.
        """
        super().__init__()
        self.__words = copy.deepcopy(words)
        self.__meta_tokens = copy.deepcopy(meta_tokens)
        self.__embedding_size = embedding_size
        self.__hidden_size = hidden_size
        self.__is_bidirectional = is_bidirectional
        self.__num_layers = num_layers
        self.__embedding = torch.nn.Embedding(
            len(words),
            embedding_size,
            padding_idx=words[meta_tokens.padding]
        )
        self.__lstm = torch.nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            bidirectional=is_bidirectional,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self,
                sentences,
                initial_context=None,
                sentences_are_sorted=False):
        """
        :param sentences: a sequence of sentences - each sentence being
        a sequence of tokens/words (strs).
        :param initial_context: a pair of torch.tensors whose shape is
        (num_layers * num_directions, batch_size, hidden_size) - the
        initial hidden and cell states for each element in the batch,
        respectively. If omitted or `None`, both tensors default to
        zeros.
        :param sentences_are_sorted: a boolean value indicating whether
        the sentences are sorted by length in decreasing order. Defaults
        to `False`.
        :returns: output, (h_n, c_n)
         - output: a torch.tensor of shape
         (max_sentence_length, batch_size, num_directions * hidden_size)
         containing the output features (h_t) from the last layer of the
         RNN, for each t.
         - (h_n, c_n): a pair of torch.tensors of shape
         (num_layers * num_directions, batch_size, hidden_size)
         containing the hidden and cell states for
         t = max_sentence_length, respectively.
        """
        h_0, c_0 = self.__zero_if_missing(initial_context,
                                          len(sentences))
        X = utils.prepare_batch(sentences,
                                self.__words,
                                self.__meta_tokens,
                                utils.device_of(self))
        E = self.__embedding(X)
        packed_output, (h_n, c_n) = \
            self.__lstm(
                torch.nn.utils.rnn.pack_padded_sequence(
                    E,
                    lengths=[len(s) for s in sentences],
                    enforce_sorted=sentences_are_sorted
                ),
                (h_0, c_0)
            )
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output
        )

        return (output, (h_n, c_n))

    def __zero_if_missing(self, initial_context, batch_size):
        target_shape = torch.Size((
            self.__num_layers * self.num_directions,
            batch_size,
            self.__hidden_size
        ))

        if (initial_context is None):
            device = utils.device_of(self)
            return (torch.zeros(target_shape, device=device),
                    torch.zeros(target_shape, device=device))
        else:
            assert all(t.shape == target_shape
                       for t in initial_context)
            return initial_context

    @property
    def embedding_size(self):
        """
        :returns: an int - the size of word embeddings.
        """
        return self.__embedding_size

    @property
    def hidden_size(self):
        """
        :returns: an int - the size of an RNN cell.
        """
        return self.__hidden_size

    @property
    def context_size(self):
        """
        :returns: an int, (num_directions * hidden_size).
        """
        return self.num_directions * self.__hidden_size

    @property
    def num_directions(self):
        """
        :returns: an int - the number of RNN directions.
        """
        return 1 + int(self.__is_bidirectional)

    @property
    def is_bidirectional(self):
        """
        :returns: a boolean value indicating whether the RNN is
        bidirectional.
        """
        return self.__is_bidirectional

    @property
    def num_layers(self):
        """
        :returns: an int - the number of RNN layers (in a single
        direction).
        """
        return self.__num_layers
