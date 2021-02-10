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
        Defaults to `True`. Note that this is only possible if the
        encoder and decoder's hidden sizes are the same; if the encoder
        is bidirectional, the final states in both directions are
        combined with sum.
        :param preprojection_nonlinearity: a unary function taking and
        returning a float. The model can be configured to apply a linear
        transformation on the final contexts, followed by a nonlinear
        function, prior to projecting the final contexts on the target
        words. This can be done by passing in the nonlinear function
        to this parameter. If this function is omitted (or `None`),
        the final contexts are not transformed at all.

        :raises ValueError: if initial binding is requested and the
        encoder's hidden size is incompatible with the decoder's
        hidden size.
        """
        super().__init__()
        self.__encoder = encoder
        self.__decoder = decoder
        self.__attention = attention
        self.__set_initial_binding_option(do_initial_binding,
                                          encoder,
                                          decoder)
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

    def __set_initial_binding_option(self,
                                     do_initial_binding,
                                     encoder,
                                     decoder):
        if (do_initial_binding and
            encoder.hidden_size != decoder.hidden_size):
            raise ValueError(
                "Initial binding is impossible with "
                "differing encoder and decoder hidden sizes!"
            )

        self.__do_initial_binding = do_initial_binding

    def forward(self, source, target):
        return 0

    def __decode(self, sentences, final_encoder_context):
        initial_context = None

        if (self.__do_initial_binding):
            h_n, c_n = (
                self.__reshaped_for_decoder(
                    self.__final_context_in(c)
                )
                for c in final_encoder_context
            )
            initial_context = (h_n, c_n)

        return self.__decoder(sentences, initial_context)

    def __reshaped_for_decoder(self, final_contexts):
        """
        Non-public utility method.

        Given the final (state/cell) context vector for each sentence on
        the last RNN layer, this function repeats the vector for each
        layer in the decoder so that it can be fed into the decoder as
        initial context vector on each layer.

        Thus, the shape changes from:
        (batch_size, hidden_size)
        to:
        (num_layers, batch_size, hidden_size)
        """
        num_layers = self.__decoder.num_layers

        return final_contexts.unsqueeze(0).repeat(num_layers, 1, 1)

    def __final_context_in(self, contexts):
        """
        Non-public utility method.

        Given the final encoder (state/cell) context vectors for each
        sentence on each RNN layer and in each direction, this
        method returns the final (state/cell) context vector for each
        sentence on the last RNN layer. If the layers are bidirectional,
        the vectors in both directions are summed element wise.

        Thus, the shape changes from:
        (num_layers * num_directions, batch_size, hidden_size)
        to:
        (batch_size, hidden_size)
        """
        num_layers = self.__encoder.num_layers
        _, batch_size, hidden_size = contexts.shape
        contexts = \
            contexts.view(num_layers, -1, batch_size, hidden_size)
        final_layer_contexts = contexts[-1, :, :, :]

        return (
            final_layer_contexts[0, :, :] + final_layer_contexts[1, :, :]
            if (self.__encoder.is_bidirectional)
            else final_layer_contexts[0, :, :]
        )

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
