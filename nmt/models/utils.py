import torch


def prepare_batch(sentences, word_indexes, meta_tokens, device):
    """
    :param sentences: a sequence of sentences - each sentence being
    a sequence of tokens/words (strs).
    :param word_indexes: a mapping with words as keys and indexes
    as values - the dictionary of the sentences' language.
    :param meta_tokens: the MetaTokens instance used when extracting
    the dictionary.
    :param device: the torch.device for the resulting tensor.
    :returns: a torch.tensor of word indexes, its shape is
    (max_sentence_len, batch_size). Each sentence is represented as
    a column, unrecognised words are replaced with the unknown token
    and the ends of the sentences are padded with the padding token
    so that they have the same length.
    """
    unknown_index, padding_index = \
        _indexes_of_unknown_and_padding_tokens_in(word_indexes,
                                                  meta_tokens)
    max_sentence_len = max(len(s) for s in sentences)
    index_sentences = ([word_indexes.get(word, unknown_index)
                        for word in s]
                       for s in sentences)
    padded_sentences = [
        s + (max_sentence_len - len(s)) * [padding_index]
        for s in index_sentences
    ]
    return torch.t(torch.tensor(padded_sentences,
                                dtype=torch.long,
                                device=device))


def _indexes_of_unknown_and_padding_tokens_in(word_indexes,
                                              meta_tokens):
    return (word_indexes[getattr(meta_tokens, name)]
            for name in ["unknown", "padding"])


def device_of(model):
    """
    :param model: an instance of type torch.mm.Module.
    :returns: the torch.device of the model's parameters, if it has any,
    and `None` otherwise.
    """
    param = next(model.parameters(), None)

    return (param.device
            if (param is not None)
            else None)


def softmax(v):
    """
    Computes the softmax function along dimension 0 in a torch.tensor.
    """
    exps = torch.exp(v)

    return exps / torch.sum(exps, dim=0).unsqueeze(0)


def flip(dictionary):
    """
    Flips a dictionary with unique values.

    :param dictionary: a mapping that is assumed to have unique values.
    :returns: a dict whose keys are the mapping's values and whose
    values are the mapping's keys.
    """
    return {
        value: key
        for key, value in dictionary.items()
    }
