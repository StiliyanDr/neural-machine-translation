from nmt.models.seqencoder import SequenceEncoder


class Encoder(SequenceEncoder):
    """
    Represents an encoder module in a sequence-to-sequence architecture.

    This module consists of two layers - an embedding followed by a RNN
    (LSTM).
    """
    def __init__(self,
                 words,
                 meta_tokens,
                 embedding_size,
                 hidden_size,
                 is_bidirectional=True,
                 num_layers=1,
                 dropout=0.):
        """
        :param words: a mapping with words (strs) for keys and indexes
        (ints) for values - the dictionary of the language being
        translated.
        :param meta_tokens: the MetaTokens instance used when extracting
        the dictionary.
        :param embedding_size: a positive integer - the size of word
        embeddings.
        :param hidden_size: a positive integer - the size of RNN cells.
        :param is_bidirectional: a boolean value indicating whether the
        RNN should be bidirectional. Defaults to `True`.
        :param num_layers: a positive integer - the number of RNN layers
        (in a single direction). Defaults to 1.
        :param dropout: a float between 0 and 1 - if non-zero,
        introduces a Dropout layer on the outputs of each RNN layer
        except the last one, with dropout probability equal to `dropout`.
        Defaults to 0.
        """
        super().__init__(words,
                         meta_tokens,
                         embedding_size,
                         hidden_size,
                         is_bidirectional,
                         num_layers,
                         dropout)
