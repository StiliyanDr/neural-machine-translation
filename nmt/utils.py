from configparser import ConfigParser
import pickle


def serialise(obj, path):
    """
    Serialises a pickleable object to a file.

    :param obj: the object to serialise.
    :param path: a str - the path to the file.
    """
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def deserialise_from(path):
    """
    Deserialises a pickleable object from a file.

    :param path: a str - the path of the file.
    :returns: the deserialised Python object.
    """
    with open(path, "rb") as file:
        return pickle.load(file)


def select_from(items, indexes):
    """
    :param items: a list of items.
    :param indexes: an iterable of indexes within `items`.
    :returns: a list of the items corresponding to the indexes.
    """
    return [items[i] for i in indexes]


def load_parameters_from(path):
    """
    Loads the parameters from an ini-style file.

    :param path: a str - the path of the file.
    :returns: a ConfigParser object.
    """
    assert isinstance(path, str)
    parser = ConfigParser()
    loaded = parser.read(path)

    if (loaded and loaded[0] == path):
        return parser
    else:
        raise IOError(
            f"Error with {path!r}!"
        )


def divides(a, b):
    """
    Given two integers `a` (!= 0) and `b`, returns a boolean value
    indicating whether a divides b.
    """
    assert a != 0

    return b % a == 0
