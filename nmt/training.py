import math
from timeit import default_timer as timer

import numpy as np
import torch

from nmt import utils


class TrainingParameters:
    """
    Bundles up model-training parameters:
     - learning_rate: a float - the (initial) learning rate.
     - learning_rate_decay: a float between 0 and 1 - when the loss
       stops improving, the learning rate is updated by multiplying it
       with this number.
     - max_gradient_norm: a float - the max norm for gradients allowed,
       used to prevent gradient explosion.
     - batch_size: a (positive) int
     - max_epochs: a (positive) int
    """
    def __init__(self,
                 learning_rate,
                 learning_rate_decay,
                 max_gradient_norm,
                 batch_size,
                 max_epochs):
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.max_gradient_norm = max_gradient_norm
        self.batch_size = batch_size
        self.max_epochs = max_epochs


class ModelTrainer:
    """
    Instances of this class are responsible for training a NMTModel.

    The model is trained by minimising the cross-entropy loss (on a
    train corpus) using stochastic gradient descent. The algorithm also
    calculates the perplexity of the model on a validation corpus; this
    happens on a given interval of parameter updates. If the perplexity
    has improved over this interval, then the model is stored on disk.
    If, however, the perplexity does not improve over a certain number
    of intervals, the learning rate is tuned and the latest model is
    picked up from disk (the one with best perplexity so far). If even
    a certain number of learning rate changes don't result in perplexity
    improvement, then the training is interrupted.
    """
    def __init__(self,
                 model_path,
                 optimizer_path,
                 log_interval=10,
                 test_interval=100,
                 max_patience=5,
                 max_trials=5):
        """
        :param model_path: a str - a path to the file where to store
        the model during training.
        :param optimizer_path: a str - a path to the file where to store
        the optimizer during training.
        :param log_interval: a positive integer - the interval of model
        updates on which to log training updates including the loss.
        Defaults to 10.
        :param test_interval: a positive integer - the interval of model
        updates on which to test whether the model's perplexity on the
        validation corpus has improved. Defaults to 100.
        :param max_patience: a positive integer - the number of
        perplexity checks with negative outcome before tuning the
        learning rate. Defaults to 5.
        :param max_trials: a positive integer - the maximum number of
        learning rate updates before the training is interrupted.
        Defaults to 5.
        """
        self.__model_path = model_path
        self.__optimizer_path = optimizer_path
        self.__log_interval = log_interval
        self.__test_interval = test_interval
        self.__max_patience = max_patience
        self.__max_trials = max_trials

    def __call__(self,
                 model,
                 optimizer,
                 train_samples,
                 validation_samples,
                 params,
                 best_perplexity=math.inf):
        """
        :param model: the instance of NMTModel to train.
        :param optimizer: the optimizer to use during training.
        :param train_samples: an instance of Samples - the training
        samples.
        :param validation_samples: an instance of Samples - the
        validation samples.
        :param params: an instance of TrainingParameters.
        :param best_perplexity: a float - the best model perplexity so
        far. If the model is yet to be trained, this value can be
        omitted and it will default to infinity.
        """
        model.train()
        trial = 0
        patience = 0
        iteration = 0
        samples_indexes = np.arange(len(train_samples.source),
                                    dtype="int32")
        begin_time = timer()

        for epoch in range(params.max_epochs):
            np.random.shuffle(samples_indexes)
            target_words = 0
            train_time = timer()

            for b in range(0, len(samples_indexes), params.batch_size):
                iteration += 1
                batch_indexes = \
                    samples_indexes[b: b + params.batch_size]
                source_batch, target_batch = \
                    self.__class__.__select_and_sort_batch(
                        batch_indexes,
                        train_samples
                    )
                target_words += sum(len(s) - 1 for s in target_batch)
                H = model(source_batch, target_batch)
                optimizer.zero_grad()
                H.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    params.max_gradient_norm
                )
                optimizer.step()

                train_time, target_words = self.__potentially_log_update(
                    iteration,
                    train_time,
                    target_words,
                    epoch,
                    b,
                    len(samples_indexes),
                    H,
                    begin_time,
                    params
                )

                new_state = self.__tune_if_there_is_no_progress(
                    model,
                    optimizer,
                    params,
                    validation_samples,
                    best_perplexity,
                    patience,
                    trial,
                    iteration
                )

                if (new_state is None):
                    return

                (model,
                 optimizer,
                 params,
                 patience,
                 trial,
                 best_perplexity) = new_state

        print("Reached maximum number of epochs!")
        self.__save_if_perplexity_is_better(model,
                                            optimizer,
                                            validation_samples,
                                            best_perplexity,
                                            params)

    @staticmethod
    def __select_and_sort_batch(batch_indexes, train_samples):
        source_batch = utils.select_from(train_samples.source,
                                         batch_indexes)
        target_batch = utils.select_from(train_samples.target,
                                         batch_indexes)
        source_and_target = sorted(
            zip(source_batch, target_batch),
            key=lambda e: len(e[0]),
            reverse=True
        )

        return tuple(zip(*source_and_target))

    def __potentially_log_update(self,
                                 iteration,
                                 train_time,
                                 target_words,
                                 epoch,
                                 batch,
                                 samples_count,
                                 H,
                                 begin_time,
                                 params):
        if (utils.divides(self.__log_interval, iteration)):
            current_time = timer()
            print("Iteration: ", iteration,
                  ", Epoch: ", epoch + 1, "/", params.max_epochs,
                  ", Batch: ", batch // params.batch_size + 1,
                  "/", samples_count // params.batch_size + 1,
                  ", Loss: ", H.item(), ", words/sec: ",
                  target_words / (current_time - train_time),
                  ", time elapsed: ", (current_time - begin_time), "s",
                  sep="")
            train_time = timer()
            target_words = 0

        return (train_time, target_words)

    def __tune_if_there_is_no_progress(self,
                                       model,
                                       optimizer,
                                       params,
                                       validation_samples,
                                       best_perplexity,
                                       patience,
                                       trial,
                                       iteration):
        if (utils.divides(self.__test_interval, iteration)):
            model.eval()
            current_perplexity = perplexity_of(
                model,
                validation_samples,
                params.batch_size
            )
            model.train()
            print("Current model perplexity:", current_perplexity)

            if (current_perplexity < best_perplexity):
                patience = 0
                best_perplexity = current_perplexity
                print("Saving new best model.")
                save_model_and_optimizer(
                    model,
                    optimizer,
                    self.__model_path,
                    self.__optimizer_path,
                    best_perplexity,
                    params.learning_rate
                )
            else:
                patience += 1

                if (patience >= self.__max_patience):
                    trial += 1

                    if (trial >= self.__max_trials):
                        print("Exceeded max trials. EARLY STOP!")
                        return None

                    params.learning_rate *= params.learning_rate_decay
                    print("Loading previously best model "
                          "and decaying learning rate to:",
                          params.learning_rate)
                    (model,
                     optimizer,
                     best_perplexity,
                     _) = \
                        load_model_and_optimizer(
                            model,
                            optimizer,
                            self.__model_path,
                            self.__optimizer_path,
                            params.learning_rate
                        )
                    patience = 0

        return (model,
                optimizer,
                params,
                patience,
                trial,
                best_perplexity)

    def __save_if_perplexity_is_better(self,
                                       model,
                                       optimizer,
                                       validation_samples,
                                       best_perplexity,
                                       params):
        model.eval()
        current_perplexity = perplexity_of(model,
                                           validation_samples,
                                           params.batch_size)
        print("Last model perplexity:", current_perplexity)

        if (current_perplexity < best_perplexity):
            print("Saving last model.")
            save_model_and_optimizer(
                model,
                optimizer,
                self.__model_path,
                self.__optimizer_path,
                current_perplexity,
                params.learning_rate
            )


def save_model_and_optimizer(model,
                             optimizer,
                             model_path,
                             optimizer_path,
                             best_perplexity,
                             learning_rate):
    model.save(model_path)
    torch.save((best_perplexity,
                learning_rate,
                optimizer.state_dict()),
               optimizer_path)


def load_model_and_optimizer(model,
                             optimizer,
                             model_path,
                             optimizer_path,
                             overriding_learning_rate=None):
    model.load(model_path)
    best_perplexity, learning_rate, osd = torch.load(
        optimizer_path
    )
    optimizer.load_state_dict(osd)

    if (overriding_learning_rate is not None):
        learning_rate = overriding_learning_rate

    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate

    return (model, optimizer, best_perplexity, learning_rate)


def perplexity_of(model, samples, batch_size):
    """
    :param model: a modelModel instance.
    :param samples: a Samples instance.
    :param batch_size: an (unsigned) int - the batch size with which
    the model was trained.
    :returns: a float - the perplexity of the model on the given
    samples.
    """
    H = 0.
    total_target_words = 0

    for b in range(0, len(samples.source), batch_size):
        source_batch = samples.source[b: b + batch_size]
        target_batch = samples.target[b: b + batch_size]
        target_words = sum(len(s) - 1 for s in target_batch)
        total_target_words += target_words

        with torch.no_grad():
            H += target_words * model(source_batch, target_batch)

    return math.exp(H / total_target_words)
