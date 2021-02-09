import torch


class Attention(torch.nn.Module):
    """
    Represents an attention module in a sequence-to-sequence
    architecture.

    Given a context from a target sentence and all the contexts from
    the corresponding source sentence, this module:
     1. Computes the attention scores for the source contexts
     2. Transforms the scores into weights by using softmax
     3. Obtains the attention vector by computing the linear combination
        of the source contexts, using the weights from 2.

    The module works with batches of sentences, not with a single pair,
    although this is also possible by constructing tensors of the
    appropriate shapes and feeding them into the module.

    The module supports two types of attention score:
     - multiplicative (the default): <W_1.h_e, W_2.h_d>
     - additive: <v, tanh(W_1.h_e + W_2.h_d)>
    where W_1, W_2 and v are module parameters and h_e and h_d are
    context vectors from the encoder and decoder, respectively.
    """
    def __init__(self,
                 source_size,
                 target_size,
                 attention_size,
                 is_multiplicative=True):
        """
        :param source_size: a positive int - the size of encoder context
        vectors.
        :param target_size: a positive int - the size of decoder context
        vectors.
        :param attention_size: a positive int - the common size to which
        both types of context vectors are transformed during attention
        score computations. That is, the number of rows in W_1 and W_2
        (as well as the size of v).
        :param is_multiplicative: a boolean value indicating whether
        the type of attention (score) is multiplicative. The other
        option is additive. Defaults to `True` so multiplicative.
        """
        super().__init__()
        self.__source_size = source_size
        self.__target_size = target_size
        self.__attention_size = attention_size
        self.__v = (None
                    if (is_multiplicative)
                    else torch.nn.Parameter(torch.rand(attention_size)))
        self.__W1 = torch.nn.Linear(source_size,
                                    attention_size,
                                    bias=False)
        self.__W2 = torch.nn.Linear(target_size,
                                    attention_size,
                                    bias=False)

    def forward(self, source, target):
        """
        :param source: a torch.tensor of shape
        (max_source_sentence_length, batch_size, source_size) containing
        the context vectors for each source sentence and each context
        length.
        :param target: a torch.tensor of shape
        (max_target_sentence_length, batch_size, target_size) containing
        the context vectors for each target sentence and each context
        length.
        :returns: a torch.tensor of shape
        (max_target_sentence_length, batch_size, source_size)
        containing the attention vectors - one for each position in each
        target sentence.
        """
        weights = self.__class__.__as_weights(
            self.__attention_scores(source, target)
        )
        # (max_source_len, max_target_len, batch_size, 1)
        extended_weights = weights.unsqueeze(3)
        # (max_source_len, 1, batch_size, source_size)
        extended_source = source.unsqueeze(1)

        return torch.sum(extended_weights * extended_source, dim=0)

    @staticmethod
    def __as_weights(attention_scores):
        """
        Non-public utility method.

        Given the attention scores, this function transforms them into
        attention weights using softmax.

        The input and output shape is:
        (max_source_len, max_target_len, batch_size)
        """
        exp_scores = torch.exp(attention_scores)

        return exp_scores / torch.sum(exp_scores, dim=0).unsqueeze(0)

    def __attention_scores(self, source, target):
        """
        Non-public utility method.

        Given the source or target context vectors, this method
        computes the attention scores.

        Thus, the shape changes from:
        (max_source_len, batch_size, source_size),
        (max_target_len, batch_size, target_size)
        to:
        (max_source_len, max_target_len, batch_size)
        """
        transformed_source = self.__transformed(source)
        transformed_target = self.__transformed(target, is_source=False)
        # (max_source_len, 1, batch_size, attention_size)
        extended_source = transformed_source.unsqueeze(1)
        # (1, max_target_len, batch_size, attention_size)
        extended_target = transformed_target.unsqueeze(0)

        return (
            torch.sum(extended_source * extended_target, dim=3)
            if (self.is_multiplicative)
            else torch.matmul(
                torch.tanh(extended_source + extended_target),
                self.__v
            )
        )

    def __transformed(self, contexts, is_source=True):
        """
        Non-public utility method.

        Given the source or target context vectors, this method
        transforms them into context vectors in the attention vector
        space with the corresponding matrix.

        Thus, the shape changes from:
        (max_len, batch_size, context_size)
        to:
        (max_len, batch_size, attention_size)
        """
        max_len, batch_size, context_size = contexts.shape
        transformation = (self.__W1 if (is_source) else self.__W2)
        transformed_contexts = transformation(contexts.flatten(0, 1))

        return transformed_contexts.view(max_len,
                                         batch_size,
                                         self.__attention_size)

    @property
    def source_size(self):
        """
        :returns: the size of source context vectors.
        """
        return self.__source_size

    @property
    def target_size(self):
        """
        :returns: the size of target context vectors.
        """
        return self.__target_size

    @property
    def attention_size(self):
        """
        :returns: the attention vector space size - the common size
        to which both types of context vectors are transformed during
        attention score computation.
        """
        return self.__attention_size

    @property
    def is_multiplicative(self):
        """
        :returns: a boolean value indicating whether the attention
        (score) type computed by the module is multiplicative. `False`
        indicates that it is additive.
        """
        return self.__v is None
