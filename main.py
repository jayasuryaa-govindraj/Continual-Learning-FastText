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

from UshurLanguageEngine.linode.data import TextData
from UshurLanguageEngine.linode.classifiers.fast_text import FastTextClassifier

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

import fasttext as ft
import time
import os
import logging
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
                    prefix = ' '.join([f"__{self.label_prefix}__{el}" for el in topic.split(self.topic_sep)])
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

            model = ft.train_supervised(input=self.train_split_file, **args, **kwargs)

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
    epoch = 117,
    lr = 0.4,
    thread=8,
    wordNgrams = 2,
    loss_fn='softmax'
)

data = pd.read_csv("askunum__820k_preprocessed.csv")
data = data.sample(frac=1)
print(data.topic.value_counts())

labels = ['Employee Coding', 'Enrollment Submission', 'Add, Remove, or Update user access', 'Enrollment Status']
data = data[data['topic'].isin(labels)]

data = data[data['topic'].isin(labels)]
train_dataset, test_dataset = train_test_split(data, test_size = 0.5)
print(len(train_dataset), len(test_dataset))

trainX = train_dataset.iloc[:30000, 1]
trainY = train_dataset.iloc[:30000, 0].values

testX = test_dataset.iloc[:30000, 1]
testY = test_dataset.iloc[:30000, 0].values

trainX = trainX.to_numpy(dtype = 'U')
testX = testX.to_numpy(dtype = 'U')

print(trainX.shape, testX.shape)

encoder = LabelEncoder()
trainY = encoder.fit_transform(trainY)
testY = encoder.transform(testY)

print(np.unique(trainY, return_counts= True), np.unique(testY, return_counts= True))

testX, validationX, testY, validationY = train_test_split(testX, testY, test_size = 1000)

ft_clf = ft_new(
    multilabel=False,
    param_search=False,
    epoch = 117,
    lr = 0.4,
    thread=8,
    wordNgrams = 2,
    loss_fn='softmax'
)

test_data = TextData(texts = testX,topics = testY)
train_data = TextData(texts = trainX,topics = trainY)

ft_clf.train(train_data,epoch=100, lr=0.25, wordNgrams=2)
pred = ft_clf.predict(test_data)
pred.preds = pred.preds.astype('int64')
print(accuracy_score(pred.preds, testY))

def capture_wrong_predictions(predictions, truth):
  # for i in range(len(testX)):
  
  wrong_indices = [i for i in range(len(predictions)) if predictions[i] != truth[i]]
  return wrong_indices