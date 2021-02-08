import math
import sys

from nltk.translate.bleu_score import corpus_bleu
import torch

from nmt import (
    datapreparation as dp,
    training as tr,
    utils,
)
from nmt.model import NMTModel


def load_parameters(path):
    params = utils.load_parameters_from(path)
    print(f"Using configuration from {path!r}.")

    tokens_params = params["META TOKENS"]
    meta_tokens = dp.MetaTokens(
        start=tokens_params["start"],
        end=tokens_params["end"],
        unknown=tokens_params["unknown"],
        padding=tokens_params["padding"]
    )

    device = torch.device(
        "cuda:0"
        if (torch.cuda.is_available())
        else "cpu"
    )
    print(f"Best device available: {device.type}.")

    return (params, meta_tokens, device)


def prepare_data(params, meta_tokens):
    """
    Loads all the samples from disk, builds objects for easier
    manipulation of the data and serialises them on disk for later
    (re)use.

    :param params: a ConfigParser - the params read from the
    'parameters.ini' file.
    :param meta_tokens: a MetaTokens instance created from the
    parameters file.
    """
    samples_params = params["SAMPLES"]
    serialisation_paths = params["SERIALISATION"]
    threshold = samples_params.getint("threshold", fallback=None)

    train_samples = dp.prepare(samples_params["train_source"],
                               samples_params["train_target"],
                               meta_tokens,
                               token_count_threshold=threshold)
    validation_samples = dp.prepare(samples_params["validation_source"],
                                    samples_params["validation_target"],
                                    meta_tokens,
                                    extract_dictionaries=False)
    test_samples = dp.prepare(samples_params["test_source"],
                              samples_params["test_target"],
                              meta_tokens,
                              extract_dictionaries=False)

    utils.serialise(train_samples,
                    serialisation_paths["train_samples"])
    utils.serialise(validation_samples,
                    serialisation_paths["validation_samples"])
    utils.serialise(test_samples,
                    serialisation_paths["test_samples"])
    utils.serialise((train_samples.source_dict,
                     train_samples.target_dict),
                    serialisation_paths["dictionaries"])


def train_model(params, device, is_extra_trained):
    """
    (Extra) trains a model.

    :param params: a ConfigParser - the params read from the
    'parameters.ini' file.
    :param device: the torch.device to use during training.
    :param is_extra_trained: a boolean value indicating whether the
    model has already been trained (so that it gets picked up from disk
    in this case).
    """
    train_samples, validation_samples = load_samples(params)
    model, optimizer, best_perplexity, learning_rate = \
        set_up_model_and_optimizer(params, device, is_extra_trained)
    trainer, training_params = set_up_trainer_and_its_params(
        params,
        learning_rate
    )
    trainer(model,
            optimizer,
            train_samples,
            validation_samples,
            training_params,
            best_perplexity)


def load_samples(params):
    return (
        utils.deserialise_from(
            params["SERIALISATION"][f"{name}_samples"]
        )
        for name in ["train", "validation"]
    )


def set_up_model_and_optimizer(params,
                               device,
                               is_extra_trained):
    model = NMTModel().to(device)
    train_params = params["TRAIN PARAMS"]
    learning_rate = train_params.getfloat("learning_rate")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    if (is_extra_trained):
        paths = params["SERIALISATION"]
        model, optimizer, best_perplexity, learning_rate = \
            tr.load_model_and_optimizer(
                model,
                optimizer,
                paths["model"],
                paths["optimizer"]
            )
    else:
        uniform_init = train_params.getfloat("uniform_init")

        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

        best_perplexity = math.inf

    return (model, optimizer, best_perplexity, learning_rate)


def set_up_trainer_and_its_params(params, learning_rate):
    train_params = params["TRAIN PARAMS"]
    paths = params["SERIALISATION"]

    trainer = tr.ModelTrainer(
        model_path=paths["model"],
        optimizer_path=paths["optimizer"],
        log_interval=train_params.getint("log_interval"),
        test_interval=train_params.getint("test_interval"),
        max_patience=train_params.getint("max_patience"),
        max_trials=train_params.getint("max_trials"),
    )
    training_params = tr.TrainingParameters(
        learning_rate=learning_rate,
        learning_rate_decay=train_params.getfloat("learning_rate_decay"),
        max_gradient_norm=train_params.getfloat("max_gradient_norm"),
        batch_size=train_params.getint("batch_size"),
        max_epochs=train_params.getint("max_epochs")
    )

    return (trainer, training_params)


def print_perplexity(params, device, samples_name):
    """
    Prints the perplexity of a model on a given set of samples.

    :param params: a ConfigParser - the params read from the
    'parameters.ini' file.
    :param device: the torch.device to use.
    :param samples_name: a str - the samples on which to compute the
    perplexity. Must be either of: 'train', 'test', 'validation'.
    """
    assert samples_name in ["train", "test", "validation"]
    paths = params["SERIALISATION"]
    samples = utils.deserialise_from(
        paths[f"{samples_name}_samples"]
    )

    model = NMTModel().to(device)
    model.load(paths["model"])

    p = tr.perplexity_of(
        model,
        samples,
        params["TRAIN PARAMS"].getint("batch_size")
    )

    print("Model perplexity:", p)


def translate(sentences_path, translation_path, device, params):
    """
    Translates a sequence of sentences in a file and stores the
    translations into another file.

    :param sentences_path: a str - the path of the file containing the
    sentences to translate.
    :param translation_path: a str - the path of the file where to write
    the translations.
    :param device: the torch.device to use.
    :param params: a ConfigParser - the params read from the
    'parameters.ini' file.
    """
    sentences = dp.read_corpus(sentences_path)
    model = NMTModel().to(device)
    model.load(params["SERIALISATION"]["model"])
    model.eval()

    with open(translation_path, "w") as file:
        for s in sentences:
            file.write(" ".join(model.translate_sentence(s)))
            file.write("\n")


def print_bleu_score(reference_path, translation_path):
    """
    Given the paths of the reference sentences and the path of the
    translated sentences, this function computes the BLEU score of the
    translation and prints it.
    """
    reference_sentences = \
        [[s] for s in dp.read_corpus(reference_path)]
    translated_sentences = \
        dp.read_corpus(translation_path)

    bleu_score = corpus_bleu(reference_sentences, translated_sentences)
    print("Corpus BLEU:", bleu_score * 100)


def write_error(message):
    sys.stderr.write(f"{message}\n")


if (__name__ == "__main__"):
    if (len(sys.argv) < 2):
        params, meta_tokens, device = load_parameters("parameters.ini")
        command_name = sys.argv[1]

        if (command_name == "prepare"):
            prepare_data(params, meta_tokens)
        elif (command_name == "train" or command_name == "extratrain"):
            train_model(
                params,
                device,
                is_extra_trained=command_name.startswith("extra")
            )
        elif (command_name == "perplexity"):
            print_perplexity(
                params,
                device,
                samples_name=("test"
                              if (len(sys.argv) == 2)
                              else sys.argv[2])
            )
        elif (command_name == "translate"):
            if (len(sys.argv) == 4):
                translate(sys.argv[2], sys.argv[3], device, params)
            else:
                write_error("Expected paths to file with sentences "
                            "and file where to store translations!")
                sys.exit(3)
        elif (command_name == "bleu"):
            if (len(sys.argv) == 4):
                print_bleu_score(sys.argv[2], sys.argv[3])
            else:
                write_error("Expected paths to reference "
                            "and machine translation!")
                sys.exit(4)
        else:
            write_error(f"No command named '{command_name}'!")
            sys.exit(2)
    else:
        write_error("Expected a command!")
        sys.exit(1)
