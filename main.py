import logging
import os
import time
import fasttext as ft
from skopt.utils import use_named_args
from skopt.space import Real, Integer
from skopt import gp_minimize
from UshurLanguageEngine.linode.classifiers.fast_text import FastTextClassifier
from UshurLanguageEngine.linode.data import TextData
import pwd
import numpy as np
from sklearn.metrics import accuracy_score
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from numpy.ma import MaskedArray
import sklearn.utils.fixes
sklearn.utils.fixes.MaskedArray = MaskedArray

# sys.path.append('/Users/jayasuryaagovindraj/Documents/Ushur_Internship/Coding/Continual-Learning-FastText/UshurLanguageEngine')


logger = logging.getLogger(__name__)

print('Hello!')


class ft_new(FastTextClassifier):
    def __init__(
        self,
        quantize=False,
        label_prefix="label",
        multilabel=False,
        param_search=False,
        search_iter=10,
        epoch_range=[1, 60],
        lr_range=[0.1, 2],
        wordNgrams_range=[2],
        loss_fn="softmax",
        topic_sep='|',
        **kwargs,
    ):
        super().__init__()

        self.model = None
        self.quantize = quantize
        self.label_prefix = label_prefix
        self.multilabel = multilabel
        self.loss_fn = loss_fn
        self.topic_sep = topic_sep

        self.param_search = param_search
        if self.param_search:
            self.init_autotune_param(
                epoch_range, lr_range, wordNgrams_range, search_iter
            )

    def write_ft_format_file(self, topics, texts):
        train_file = self._create_tmp_train_fname()

        with open(train_file, "w") as f:
            for topic, text in zip(topics, texts):
                if self.multilabel:
                    prefix = ' '.join(
                        [f"__{self.label_prefix}__{el}" for el in topic.split(self.topic_sep)])
                    f.write(f"{prefix} {text}\n")
                else:
                    f.write(f"__{self.label_prefix}__{topic} {text}\n")

        return train_file

    def train(
        self, data, epoch=50, lr=0.1, thread=1, wordNgrams=1, **kwargs,
    ):

        if self.param_search:
            optimal_param = self.autotune(data)
            logger.info("optimal parameters:")
            logger.info(optimal_param)

        train_file = self.write_ft_format_file(data.topics, data.texts)

        loss = "ova" if self.multilabel else self.loss_fn
        logger.info(f"FastText Model Loss Function is {loss}")

        if self.param_search:
            self.model = ft.train_supervised(
                input=train_file, loss=loss, **optimal_param, **kwargs,
            )
        else:
            self.model = ft.train_supervised(
                input=train_file,
                loss=loss,
                epoch=epoch,
                lr=lr,
                thread=thread,
                wordNgrams=wordNgrams,
                **kwargs,
            )

    def autotune(self, data, **kwargs):
        logger.info("preparing autotune data")
        # train val split
        train_split, val_split = data.train_test_split(0.2)
        # split to ft file
        self.train_split_file = self.write_ft_format_file(
            train_split.topics, train_split.texts
        )
        self.val_split_file = self.write_ft_format_file(
            val_split.topics, val_split.texts
        )

        @use_named_args(self.search_space)
        def autotune_objective(**params):
            args = self.fixed_param.copy()
            args.update(params)

            args["loss"] = "ova" if self.multilabel else self.loss_fn

            model = ft.train_supervised(
                input=self.train_split_file, **args, **kwargs)

            val_result = model.test(self.val_split_file)

            return 1 - np.mean(val_result[1:])

        logger.info("starting autotune")
        gstart = time.time()

        res_gp = gp_minimize(
            autotune_objective, self.search_space, n_calls=self.search_iter
        )

        logger.info("Best score=%.4f" % res_gp.fun)
        gend = time.time()
        logger.info("autotune time %f" % ((gend - gstart) / 60))

        logger.info("cleaning up autotune")
        os.remove(self.train_split_file)
        del self.train_split_file
        os.remove(self.val_split_file)
        del self.val_split_file

        logger.info("autotune done")
        logger.info(res_gp)

        optimal_param = dict(zip(self.search_params, res_gp.x))
        optimal_param.update(self.fixed_param)
        return optimal_param


ft_clf = ft_new(
    multilabel=False,
    param_search=False,
    epoch=117,
    lr=0.4,
    thread=8,
    wordNgrams=2,
    loss_fn='softmax'
)

data = pd.read_csv("askunum__820k_preprocessed.csv")
data = data.sample(frac=1)
print(data.topic.value_counts())

labels = ['Employee Coding', 'Enrollment Submission',
          'Add, Remove, or Update user access', 'Enrollment Status']
data = data[data['topic'].isin(labels)]

data = data[data['topic'].isin(labels)]
train_dataset, test_dataset = train_test_split(data, test_size=0.5)
print(len(train_dataset), len(test_dataset))

trainX = train_dataset.iloc[:30000, 1]
trainY = train_dataset.iloc[:30000, 0].values

testX = test_dataset.iloc[:30000, 1]
testY = test_dataset.iloc[:30000, 0].values

trainX = trainX.to_numpy(dtype='U')
testX = testX.to_numpy(dtype='U')

print(trainX.shape, testX.shape)

encoder = LabelEncoder()
trainY = encoder.fit_transform(trainY)
testY = encoder.transform(testY)

print(np.unique(trainY, return_counts=True),
      np.unique(testY, return_counts=True))

testX, validationX, testY, validationY = train_test_split(
    testX, testY, test_size=1000)

ft_clf = ft_new(
    multilabel=False,
    param_search=False,
    epoch=117,
    lr=0.4,
    thread=8,
    wordNgrams=2,
    loss_fn='softmax'
)

test_data = TextData(texts=testX, topics=testY)
train_data = TextData(texts=trainX, topics=trainY)

ft_clf.train(train_data, epoch=100, lr=0.25, wordNgrams=2)
pred = ft_clf.predict(test_data)
pred.preds = pred.preds.astype('int64')
print(accuracy_score(pred.preds, testY))


def capture_wrong_predictions(predictions, truth):
    wrong_indices = [i for i in range(
        len(predictions)) if predictions[i] != truth[i]]
    return wrong_indices


class ModelWithFeedback:

    def __init__(self, model):
        self.model = model
        self.buffer = [[], []]
        # self.partialModel = keras.Model(inputs = self.model.inputs, outputs = self.model.layers[-2].output)
        self.tfidfEncoder = TfidfVectorizer(min_df=1, stop_words="english")

    def tfidf_encoder_fit(self, train_texts):
        self.tfidfEncoder.fit(train_texts)

    def predictV2(self, testX, threshold=0.9, verbose=False, debug=False):
        encoded_text = self.tfidfEncoder.transform(testX)
        similarities = cosine_similarity(encoded_text, self.buffer[0])

        if debug:
            return similarities

        if similarities.max() > threshold:
            if verbose:
                print("Using the feedback buffer")
                print(f"Value is at {similarities.argmax()}")
                # print(feedbackModel.buffer[1].shape)
                # print(feedbackModel.buffer[1][similarities.argmax()])
                # print(feedbackModel.buffer[1][similarities.argmax()].shape)
            return np.array([self.buffer[1][similarities.argmax()]]), True

        else:
            if verbose:
                print("Using the pure model")
            # testX_encoded = self.encode_texts(testX)
            # testX_encoded = np.pad(testX_encoded, ((0,0), (0, trainX_encoded.shape[1] - testX_encoded.shape[1])), mode = 'constant')
            test_data = TextData(texts=testX, topics=testY)
            predictions = self.model.predict(test_data)
            return np.array([predictions.preds.astype('int64')]), False
            # return np.array(self.model.predict(testX)), False

    def predict(self, testX, threshold=0.9, verbose=False, debug=False):
        partialOutput = self.partialModel.predict(testX)
        # print(partialOutput.shape)
        np_buffer = np.array(self.buffer[0])
        np_buffer = np_buffer.reshape((np_buffer.shape[0], np_buffer.shape[2]))

        similarity = cosine_similarity(partialOutput, np_buffer)

        if debug:
            print(similarity.shape)
            return np.array(self.model.predict(testX)), np.array([self.buffer[1][similarity.argmax()]])

        if similarity.max() > threshold:
            if verbose:
                print("Using the feedback buffer")
                print(f"Value is at {similarity.argmax()}")
            # print(f"Feedback buffer shape: {(self.buffer[1][similarity.argmax()]).shape}")
            # print(f"Pure model shape: {(np.array([self.model.predict(testX).argmax()])).shape}")
            return np.array([self.buffer[1][similarity.argmax()]]), True

        else:
            if verbose:
                print("Using the pure model")
            # print((np.array([self.model.predict(testX).argmax()])).shape)
            return np.array(self.model.predict(testX)), False

# Fix


def evaluate_predictionsV2(testX, testY, feedbackModel: ModelWithFeedback, test_indices, threshold=0.9, verbose=False, debug=False):
    """Function to evaluate the predictions made by the model with the true class labels.

    Args:
        testX (Numpy Array): Numpy array containing the input texts to be fed to the model.
        testY (Numpy Array): Numpy array containing the corresponding classes of the texts.
        feedbackModel (ModelWithFeedback): The feedback model used to make the prediction.
        test_indices (list): A list containing the indices of the dataset to send to the model.
        threshold (float): The threshold which decides whether the prediction is made using the baseline model or the feedback buffer. Defaults to 0.9.
        verbose (bool, optional): Provide more information about the running of the model. Defaults to False.
        debug (bool, optional): Used for debugging purposes. Code written in the debug module will run. Defaults to False.

    Returns:
        tuple: Returns the accuracy score and the number of samples correctly classified.
    """

    feedback_predictions = []
    actual_output = []
    feedbackCount = 0

    feedbackCorrect = 0
    pureModelCorrect = 0

    pure_indices = []

    for i in range(len(test_indices)):
        x_test = np.array([testX[test_indices[i]]])
        y_test = np.array([testY[test_indices[i]]])

        if debug:
            pass
            # x_test_encoded = feedbackModel.encode_texts(x_test)
            # if model.predict(x_test_encoded) == y_test:
            #   print(f'Error at {test_indices[i]}')
            break

        output, feedbackUsed = feedbackModel.predictV2(
            x_test, threshold, verbose)
        # output_bool = output
        feedback_predictions.append(output)
        actual_output.append(y_test)

        # if debug:
        #   x_test_encoded = feedbackModel.encode_texts(x_test)
        #   if model.predict(x_test_encoded) == y_test:
        #     print(f'Error at {test_indices[i]}')
        #     break

        # if debug and not feedbackUsed:
        #   print(output_bool, y_test)
        #   if output_bool == y_test:
        #     print('yes')
        #   else:
        #     print('no')

        if feedbackUsed:
            feedbackCount += 1

        # print(output, y_test)
        if output == y_test:
            # print('yes')
            if feedbackUsed:
                feedbackCorrect += 1
            else:
                pureModelCorrect += 1

        if debug:
            return feedback_predictions, actual_output

    feedback_predictions = np.array(feedback_predictions).flatten()
    actual_output = np.array(actual_output).flatten()

    print(f"The feedback buffer predicted {feedbackCorrect} samples correctly")
    print(f"The pure model predicted {pureModelCorrect} samples correctly")

    print(f"The feedback buffer was used {feedbackCount} times")
    print(f"The pure model was used {len(test_indices) - feedbackCount} times")

    if debug:
        return pure_indices

    return accuracy_score(feedback_predictions, actual_output), accuracy_score(feedback_predictions, actual_output, normalize=False)


def load_bufferV2(indices, feedbackModel: ModelWithFeedback, testX, testY):
    """Function to load the feedback buffer with the misclassified texts along with their corresponding classes. The TF-IDF representations of the texts are stored in the buffer.

    Args:
        indices (list): A list containing the indices wherein the misclassified samples are in the dataset.
        feedbackModel (ModelWithFeedback): The feedback model containing the buffer to which samples will be added to.
        testX (Numpy Array): Numpy array containing the misclassified texts to be fed to the feedback buffer.
        testY (Numpy Array): Numpy array containing the corresponding classes of the texts.
    """

    wrongX = testX[indices]
    wrongY = testY[indices]
    wrongY = wrongY.reshape((wrongY.shape[0], 1))
    encoded_texts = feedbackModel.tfidfEncoder.transform(wrongX)
    feedbackModel.buffer[0] = encoded_texts.toarray()
    feedbackModel.buffer[1] = wrongY

    return


def get_random_indices(wrong_indices, sample_size) -> tuple:
    """Function to grab random indices from a list given a sample size

    Args:
        wrong_indices (list): list containing the indices to sample from.
        sample_size (int): sample size to sample the original list from.

    Returns:
        tuple: Returns two lists- one subsampled list, and the other is the remainder.
    """
    buffer_indices = random.sample(wrong_indices, sample_size)
    test_indices = list(set(wrong_indices) ^ set(buffer_indices))
    return buffer_indices, test_indices


# Instantiate the feedback model
feedbackModel = ModelWithFeedback(ft_clf)

wrong_indices = capture_wrong_predictions(pred.preds, testY)
buffer_indices, test_indices = get_random_indices(wrong_indices, 800)
feedbackModel.tfidf_encoder_fit(trainX)

#Loading the misclassified samples into the buffer
load_bufferV2(buffer_indices, feedbackModel, testX, testY)
print(len(feedbackModel.buffer[0]))
print(len(wrong_indices), len(test_indices))

new_testX = np.concatenate((validationX, testX[test_indices]))
new_testY = np.concatenate((validationY, testY[test_indices]))

#Generating indices for the new test set
new_test_indices = [i for i in range(0, len(new_testY))]
print(len(new_test_indices))

evaluate_predictionsV2(testX, testY, test_indices[:3000], threshold = 0.9)

#Grabbing random samples from the test data
new_test_indices = [i for i in range(0, len(testY))]
new_test_indices, _ = get_random_indices(new_test_indices, 2000)

evaluate_predictionsV2(testX, testY, new_test_indices, threshold = 0.9)

#Creating a numpy array to check the performance of the baseline model with the new test data
pureX = testX[new_test_indices]
pureY = testY[new_test_indices]

pure_data = TextData(texts = pureX,topics = pureY)
pure_pred = feedbackModel.model.predict(pure_data)
pure_pred.preds = pure_pred.preds.astype('int64')
print(accuracy_score(pure_pred.preds, pureY))

#Creating indices for the validation dataset
new_test_indices = [i for i in range(0, len(validationY))]
evaluate_predictionsV2(validationX, validationY, new_test_indices, threshold = 0.9)