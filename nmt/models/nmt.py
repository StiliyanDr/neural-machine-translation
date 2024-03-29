import copy

import torch

from nmt.models import utils
from nmt.utils import default_if_none


class NMTModel(torch.nn.Module):
    """
    Represents a sequence-to-sequence neural machine translation model
    with (RNN) encoder-decoder architecture including attention.

    Some of the architecture's specifics are:
     - source sentences are fed into the encoder which returns the
       different context vectors for each sentence
     - target sentences (and optionally the transformed final state of
       the encoder) are fed into the decoder which returns the different
       context vectors for each sentence
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
                 meta_tokens,
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
        :param meta_tokens: the MetaTokens instance used when extracting
        the dictionary.
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
        final_context_size = encoder.context_size + decoder.context_size
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
        self.__set_words(target_words, meta_tokens)

    def __set_initial_binding_option(self,
                                     do_initial_binding,
                                     encoder,
                                     decoder):
        self.__initial_binding_transformation = (
            torch.nn.Linear(encoder.context_size,
                            decoder.hidden_size)
            if (do_initial_binding)
            else None
        )

    def __set_words(self, target_words, meta_tokens):
        self.__target_words = copy.deepcopy(target_words)
        self.__index_to_word = utils.flip(target_words)
        self.__meta_tokens = copy.deepcopy(meta_tokens)

    def forward(self, source, target, source_is_sorted=False):
        """
        :param source: a sequence of sentences - each sentence being
        a sequence of tokens/words (strs); the sentences being
        translated.
        :param target: a sequence of sentences - each sentence being
        a sequence of tokens/words (strs) which starts and ends with
        the start and end meta tokens, respectively; the translation
        sentences.
        :param source_is_sorted: a boolean value indicating whether the
        source sentences are sorted by length in decreasing order.
        Defaults to `False`.
        :returns: a float - the cross-entropy loss computed on the
        batch of sentences.

        Note that the two sequences are assumed to have the same length,
        each sentence in `target` corresponds to a sentence in `source`.
        """
        assert len(source) == len(target)
        source_contexts, final_encoder_context = self.__encoder(
            source,
            sentences_are_sorted=source_is_sorted
        )
        target_contexts, _ = self.__decode(
            sentences=[s[:-1] for s in target],
            final_encoder_context=final_encoder_context
        )
        attention_vectors = \
            self.__attention(source_contexts, target_contexts)
        final_contexts = \
            self.__final_contexts(target_contexts, attention_vectors)
        projections = self.__words_projection(final_contexts)
        target_words = self.__target_words_for(target)

        return self.__cross_entropy_loss(projections, target_words)

    def __decode(self, sentences, final_encoder_context):
        """
        Non-public utility method.

        Given the target sentences for the decoder and the final encoder
        state, this method feeds the sentences into the decoder,
        optionally providing it with the final encoder state (if the
        model was set up with initial binding).
        """
        return self.__decoder(
            sentences,
            self.__initial_decoder_context(final_encoder_context)
        )

    def __initial_decoder_context(self, final_encoder_context):
        initial_context = None

        if (self.__do_initial_binding()):
            h_n, c_n = (
                self.__transformed_for_decoder(
                    self.__final_context_in(c)
                )
                for c in final_encoder_context
            )
            initial_context = (h_n, c_n)

        return initial_context

    def __do_initial_binding(self):
        return self.__initial_binding_transformation is not None

    def __transformed_for_decoder(self, final_contexts):
        """
        Non-public utility method.

        Given the final (state/cell) context vector for each sentence on
        the last RNN layer, this function transforms the vector and
        repeats the resulting vector for each layer in the decoder so
        that it can be fed into the decoder as initial context vector on
        each layer.

        Thus, the shape changes from:
        (batch_size, encoder_context_size)
        to:
        (num_layers, batch_size, decoder_hidden_size)
        """
        final_contexts = self.__initial_binding_transformation(
            final_contexts
        )
        num_layers = self.__decoder.num_layers

        return final_contexts.unsqueeze(0).repeat(num_layers, 1, 1)

    def __final_context_in(self, contexts):
        """
        Non-public utility method.

        Given the final encoder (state/cell) context vectors for each
        sentence on each RNN layer and in each direction, this
        method returns the final (state/cell) context vector for each
        sentence on the last RNN layer. If the layers are bidirectional,
        the vectors in both directions are concatenated.

        Thus, the shape changes from:
        (num_layers * num_directions, batch_size, hidden_size)
        to:
        (batch_size, encoder_context_size)
        """
        num_layers = self.__encoder.num_layers
        _, batch_size, hidden_size = contexts.shape
        contexts = \
            contexts.view(num_layers, -1, batch_size, hidden_size)
        final_layer_contexts = contexts[-1, :, :, :]

        return (
            torch.cat((final_layer_contexts[0, :, :],
                       final_layer_contexts[1, :, :]),
                      dim=1)
            if (self.__encoder.is_bidirectional)
            else final_layer_contexts[0, :, :]
        )

    def __final_contexts(self, target_contexts, attention_vectors):
        """
        Non-public utility method.

        Given the target context vectors and their attention vectors,
        for each target sentence and each context in it, this method
        concatenates the context and attention vectors, optionally
        transforms the concatenated vectors (if the model was set up
        like so) and returns them.

        The shapes change from:
        (max_sent_len, batch_size, hidden_size),
        (max_sent_len, batch_size, source_size)
        to:
        (max_sent_len * batch_size, hidden_size + source_size)
        """
        final_contexts = \
            torch.cat((target_contexts, attention_vectors), dim=2)
        final_contexts = final_contexts.flatten(0, 1)

        return (
            self.__nonlinearity(self.__preprojection_transformation(
                final_contexts
            ))
            if (self.__transform_contexts_before_projection())
            else final_contexts
        )

    def __transform_contexts_before_projection(self):
        return self.__preprojection_transformation is not None

    def __target_words_for(self, sentences):
        """
        Non-public utility method.

        Given the target sentences, this method returns a tensor of
        shape ((max_sentence_len - 1) * batch_size) containing the
        indexes of the target words in each sentence. First come the
        first target words in each sentence, then the second and so on.
        """
        indexes = utils.prepare_batch(
            sentences=[s[1:] for s in sentences],
            word_indexes=self.__target_words,
            meta_tokens=self.__meta_tokens,
            device=utils.device_of(self)
        )

        return indexes.flatten(0, 1)

    def __cross_entropy_loss(self, projections, target_words):
        return torch.nn.functional.cross_entropy(
            projections,
            target_words,
            ignore_index=self.__target_words[
                self.__meta_tokens.padding
            ]
        )

    def translate_sentence(self, s, limit=None):
        """
        :param s: a sequence of tokens (strs) - the sentence to
        translate.
        :param limit: an unsigned int - a limit for the translation's
        length in tokens. If omitted or `None`, defaults to
        `TRANSLATION_LIMIT`.
        :returns: a list of tokens - the translation.
        """
        if (not s):
            return []

        with torch.no_grad():
            return self.__do_translate_sentence(
                s,
                default_if_none(
                    limit,
                    self.__class__.__TRANSLATION_LIMIT
                )
            )

    def __do_translate_sentence(self, s, limit):
        sentence_contexts, decoder_context = self.__encode_sentence(s)
        translation = []
        word = self.__meta_tokens.start

        while (len(translation) < limit):
            probabilities, decoder_context = self.__distribution_after(
                word,
                decoder_context,
                sentence_contexts
            )
            word = self.__most_probable_word(probabilities)

            if (word != self.__meta_tokens.end):
                translation.append(word)
            else:
                break

        return translation

    def __encode_sentence(self, s):
        """
        Non-public utility method.

        Computes the source contexts of shape
        (sentence_length, 1, num_directions * hidden_size)

        and the initial decoder context - a pair of tensors of shape
        (decoder_num_layers, 1, decoder_hidden_size). If initial binding
        is not used, the initial context is `None`, that is, the decoder
        will use tensors with zeros instead.
        """
        sentence_contexts, final_encoder_context = \
            self.__encoder([s], sentences_are_sorted=True)
        decoder_context = self.__initial_decoder_context(
            final_encoder_context
        )

        return (sentence_contexts, decoder_context)

    def __distribution_after(self,
                             word,
                             decoder_context,
                             sentence_contexts):
        """
        Non-public utility method.

        Given the source sentence contexts, the current state of the
        decoder (which captures the translated sentence so far) and
        the newly generated word, this method updates the decoder state,
        computes the distribution for the updated translation and
        returns the distribution and decoder state.
        """
        _, decoder_context = self.__decoder([[word]], decoder_context)
        h_n, _ = decoder_context
        target_context = h_n[-1, :, :].unsqueeze(0)
        attention_vector = self.__attention(sentence_contexts,
                                            target_context)
        final_context = \
            self.__final_contexts(target_context,
                                  attention_vector)
        projection = self.__words_projection(final_context)
        distribution = utils.softmax(projection[0, :])

        return (distribution, decoder_context)

    def __most_probable_word(self, probabilities):
        index = torch.argmax(probabilities).item()
        return self.__index_to_word[index]

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
