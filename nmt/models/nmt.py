import copy

import torch


class NMTModel(torch.nn.Module):
    """
    Represents a sequence-to-sequence neural machine translation model
    with (RNN) encoder-decoder architecture including attention.

    Some of the architecture's specifics are:
     - source sentences are fed into the encoder which returns the
       different context vectors for each sentence
     - target sentences (and optionally the final state of the encoder)
       are fed into the decoder which returns the different context
       vectors for each sentence
     - source and target context vectors are fed into the attention
       module which returns the attention vector for each target
       context vector
     - target context vectors and attention vectors are concatenated
       in order to form the final context vectors
     - the final context vectors are optionally transformed and after
       that they are projected on the target words/terms
     - the projections are used to compute the cross-entropy loss
       during training and (softmax) probabilities during translation
    """

    __TRANSLATION_LIMIT = 1000

    def __init__(self,
                 encoder,
                 decoder,
                 attention,
                 target_words,
                 do_initial_binding=True,
                 preprojection_nonlinearity=None):
        """
        :param encoder: any encoder module defined within `nmt.models`.
        :param decoder: any decoder module defined within `nmt.models`.
        :param attention: any attention module defined within
        `nmt.models`.
        :param target_words: a mapping with words (strs) for keys and
        indexes (ints) for values - the dictionary of the language being
        translated to.
        :param do_initial_binding: a boolean value indicating whether
        the final state of the encoder should be fed into the decoder.
        Defaults to `True`.
        :param preprojection_nonlinearity: a unary function taking and
        returning a float. The model can be configured to apply a linear
        transformation on the final contexts, followed by a nonlinear
        function, prior to projecting the final contexts on the target
        words. This can be done by passing in the nonlinear function
        to this parameter. If this function is omitted (or `None`),
        the final contexts are not transformed at all.
        """
        super().__init__()
        self.__encoder = encoder
        self.__decoder = decoder
        self.__attention = attention
        self.__do_initial_binding = do_initial_binding
        final_context_size = encoder.hidden_size + decoder.hidden_size
        self.__preprojection_transformation = (
            None
            if (preprojection_nonlinearity is None)
            else torch.nn.Linear(final_context_size,
                                 final_context_size)
        )
        self.__nonlinearity = preprojection_nonlinearity
        self.__words_projection = torch.nn.Linear(
            final_context_size,
            len(target_words)
        )
        self.__target_words = copy.deepcopy(target_words)

    def forward(self, source, target):
        return 0

    def translate_sentence(self, sentence, limit=None):
        return []

    def save(self, path):
        """
        Saves the state of the model on disk.

        :param path: a str - the path of the file where to save the
        model.
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Loads the state of the model from disk.

        :param path: a str - the path of the file from where to load the
        model's state.
        """
        self.load_state_dict(torch.load(path))
