'''
this code is adapted from 
https://github.com/lasso-net/lassonet/blob/master/experiments/data_utils.py
it is modified with local path of datasets. 
https://github.com/google-research/google-research/blob/master/sequential_attention/sequential_attention/experiments/datasets/data_loader.py
for activity data
'''
import pickle
from typing import Tuple, Optional
from collections import defaultdict
from os.path import join
from pathlib import Path
import sys 
import pandas as pd 

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_openml

_tf = None

def _get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf
        _tf = tf
    return _tf

# The code to load some of these datasets is reproduced from
# https://github.com/mfbalin/Concrete-Autoencoders/blob/master/experiments/generate_comparison_figures.py

def load_mice(one_hot=False):
    filling_value = -100000
    X = np.genfromtxt(
        "/Users/amber/Desktop/ece695_final_proj/datasets/MICE/Data_Cortex_Nuclear.csv",
        delimiter=",",
        skip_header=1,
        usecols=range(1, 78),
        filling_values=filling_value,
        encoding="UTF-8",
    )
    classes = np.genfromtxt(
        "/Users/amber/Desktop/ece695_final_proj/datasets/MICE/Data_Cortex_Nuclear.csv",
        delimiter=",",
        skip_header=1,
        usecols=range(78, 81),
        dtype=None,
        encoding="UTF-8",
    )

    for i, row in enumerate(X):
        for j, val in enumerate(row):
            if val == filling_value:
                X[i, j] = np.mean(
                    [
                        X[k, j]
                        for k in range(classes.shape[0])
                        if np.all(classes[i] == classes[k])
                    ]
                )

    DY = np.zeros((classes.shape[0]), dtype=np.uint8)
    for i, row in enumerate(classes):
        for j, (val, label) in enumerate(zip(row, ["Control", "Memantine", "C/S"])):
            DY[i] += (2**j) * (val == label)

    Y = np.zeros((DY.shape[0], np.unique(DY).shape[0]))
    for idx, val in enumerate(DY):
        Y[idx, val] = 1

    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    DY = DY[indices]
    classes = classes[indices]

    if not one_hot:
        Y = DY

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    print("X shape: {}, Y shape: {}".format(X.shape, Y.shape))

    return (X[: X.shape[0] * 4 // 5], Y[: X.shape[0] * 4 // 5]), (
        X[X.shape[0] * 4 // 5 :],
        Y[X.shape[0] * 4 // 5 :],
    )

def load_isolet():
    x_train = np.genfromtxt(
        "/Users/amber/Desktop/ece695_final_proj/datasets/isolet/isolet1+2+3+4.data",
        delimiter=",",
        usecols=range(0, 617),
        encoding="UTF-8",
    )
    y_train = np.genfromtxt(
        "/Users/amber/Desktop/ece695_final_proj/datasets/isolet/isolet1+2+3+4.data",
        delimiter=",",
        usecols=[617],
        encoding="UTF-8",
    )
    x_test = np.genfromtxt(
        "/Users/amber/Desktop/ece695_final_proj/datasets/isolet/isolet5.data",
        delimiter=",",
        usecols=range(0, 617),
        encoding="UTF-8",
    )
    y_test = np.genfromtxt(
        "/Users/amber/Desktop/ece695_final_proj/datasets/isolet/isolet5.data",
        delimiter=",",
        usecols=[617],
        encoding="UTF-8",
    )

    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        np.concatenate((x_train, x_test))
    )
    x_train = X[: len(y_train)]
    x_test = X[len(y_train) :]

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    return (x_train, y_train - 1), (x_test, y_test - 1)


import numpy as np

def load_epileptic():
    filling_value = -100000

    X = np.genfromtxt(
        "/Users/amber/Desktop/ece695_final_proj/datasets/data.csv",
        delimiter=",",
        skip_header=1,
        usecols=range(1, 179),
        filling_values=filling_value,
        encoding="UTF-8",
    )
    Y = np.genfromtxt(
        "/homelemisma/datasets/data.csv",
        delimiter=",",
        skip_header=1,
        usecols=range(179, 180),
        encoding="UTF-8",
    )

    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    print(X.shape, Y.shape)

    return (X[:8000], Y[:8000]), (X[8000:], Y[8000:])

import os

from PIL import Image


def load_coil():
    samples = []
    for i in range(1, 21):
        for image_index in range(72):
            obj_img = Image.open(
                os.path.join(
                    "/Users/amber/Desktop/ece695_final_proj/datasets/coil-20-proc",
                    "obj%d__%d.png" % (i, image_index),
                )
            )
            rescaled = obj_img.resize((20, 20))
            pixels_values = [float(x) for x in list(rescaled.getdata())]
            sample = np.array(pixels_values + [i])
            samples.append(sample)
    samples = np.array(samples)
    np.random.shuffle(samples)
    data = samples[:, :-1]
    targets = (samples[:, -1] + 0.5).astype(np.int64)
    data = (data - data.min()) / (data.max() - data.min())

    l = data.shape[0] * 4 // 5
    train = (data[:l], targets[:l] - 1)
    test = (data[l:], targets[l:] - 1)
    print(train[0].shape, train[1].shape)
    print(test[0].shape, test[1].shape)
    return train, test


import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_data(fashion=False, digit=None, normalize=False):
    tf = _get_tf() 
    
    if fashion:
        (x_train, y_train), (x_test, y_test) = (
            tf.keras.datasets.fashion_mnist.load_data()
        )
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if digit is not None and 0 <= digit and digit <= 9:
        train = test = {y: [] for y in range(10)}
        for x, y in zip(x_train, y_train):
            train[y].append(x)
        for x, y in zip(x_test, y_test):
            test[y].append(x)

        for y in range(10):

            train[y] = np.asarray(train[y])
            test[y] = np.asarray(test[y])

        x_train = train[digit]
        x_test = test[digit]

    x_train = x_train.reshape((-1, x_train.shape[1] * x_train.shape[2])).astype(
        np.float32
    )
    x_test = x_test.reshape((-1, x_test.shape[1] * x_test.shape[2])).astype(np.float32)

    if normalize:
        X = np.concatenate((x_train, x_test))
        X = (X - X.min()) / (X.max() - X.min())
        x_train = X[: len(y_train)]
        x_test = X[len(y_train) :]

    #     print(x_train.shape, y_train.shape)
    #     print(x_test.shape, y_test.shape)
    return (x_train, y_train), (x_test, y_test)


def load_mnist():
    train, test = load_data(fashion=False, normalize=True)
    x_train, x_test, y_train, y_test = train_test_split(test[0], test[1], test_size=0.2)
    return (x_train, y_train), (x_test, y_test)


def load_fashion():
    train, test = load_data(fashion=True, normalize=True)
    x_train, x_test, y_train, y_test = train_test_split(test[0], test[1], test_size=0.2)
    return (x_train, y_train), (x_test, y_test)


def load_mnist_two_digits(digit1, digit2):
    train_digit_1, _ = load_data(digit=digit1)
    train_digit_2, _ = load_data(digit=digit2)

    X_train_1, X_test_1 = train_test_split(train_digit_1[0], test_size=0.6)
    X_train_2, X_test_2 = train_test_split(train_digit_2[0], test_size=0.6)

    X_train = np.concatenate((X_train_1, X_train_2))
    y_train = np.array([0] * X_train_1.shape[0] + [1] * X_train_2.shape[0])
    shuffled_idx = np.random.permutation(X_train.shape[0])
    np.take(X_train, shuffled_idx, axis=0, out=X_train)
    np.take(y_train, shuffled_idx, axis=0, out=y_train)

    X_test = np.concatenate((X_test_1, X_test_2))
    y_test = np.array([0] * X_test_1.shape[0] + [1] * X_test_2.shape[0])
    shuffled_idx = np.random.permutation(X_test.shape[0])
    np.take(X_test, shuffled_idx, axis=0, out=X_test)
    np.take(y_test, shuffled_idx, axis=0, out=y_test)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    return (X_train, y_train), (X_test, y_test)


import os

from sklearn.preprocessing import MinMaxScaler

'''
def load_activity():
    x_train = np.loadtxt(
        os.path.join("/Users/amber/Desktop/ece695_final_proj/datasets/dataset_uci", "final_X_train.txt"),
        delimiter=' ', # delimiter=",",
        dtype=np.float64, # encoding="UTF-8",
    )
    print(f"x_train is {x_train.shape}")
    x_test = np.loadtxt(
        os.path.join("/Users/amber/Desktop/ece695_final_proj/datasets/dataset_uci", "final_X_test.txt"),
        delimiter=",",
        encoding="UTF-8",
    )
    
    print(f"x_test is {x_test.shape}")
    y_train = (
        np.loadtxt(
            os.path.join("/Users/amber/Desktop/ece695_final_proj/datasets/dataset_uci", "final_y_train.txt"),
            delimiter=",",
            encoding="UTF-8",
        )
        - 1
    )
    print(f"y_train is {y_train.shape}")
    y_test = (
        np.loadtxt(
            os.path.join("/Users/amber/Desktop/ece695_final_proj/datasets/dataset_uci", "final_y_test.txt"),
            delimiter=",",
            encoding="UTF-8",
        )
        - 1
    )
    print(f"y_test is {y_test.shape}")

    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        np.concatenate((x_train, x_test))
    )
    x_train = X[: len(y_train)]
    x_test = X[len(y_train) :]

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    return (x_train, y_train), (x_test, y_test)
'''

def load_activity():
  """Loads the Activity dataset, adapted from: https://github.com/lasso-net/lassonet/blob/master/experiments/data_utils.py."""
  DATA_DIR = "/Users/amber/Desktop/ece695_proj/src/sequential_attention/experiments/datasets"
  cache_filepath_train_x = os.path.join(DATA_DIR, "activity/X_train.txt")
  cache_filepath_train_y = os.path.join(DATA_DIR, "activity/y_train.txt")
  cache_filepath_test_x = os.path.join(DATA_DIR, "activity/X_test.txt")
  cache_filepath_test_y = os.path.join(DATA_DIR, "activity/y_test.txt")
  with open(cache_filepath_train_x, "r") as fp:
    x_train = np.genfromtxt(fp.readlines(), encoding="UTF-8")
  with open(cache_filepath_test_x, "r") as fp:
    x_test = np.genfromtxt(fp.readlines(), encoding="UTF-8")
  with open(cache_filepath_train_y, "r") as fp:
    y_train = np.genfromtxt(fp.readlines(), encoding="UTF-8")
  with open(cache_filepath_test_y, "r") as fp:
    y_test = np.genfromtxt(fp.readlines(), encoding="UTF-8")

  x = MinMaxScaler(feature_range=(0, 1)).fit_transform(
      np.concatenate((x_train, x_test))
  )
  x_train = x[: len(y_train)]
  x_test = x[len(y_train) :]

  print("Data loaded...")
  print("Data shapes:")
  print(x_train.shape, y_train.shape)
  print(x_test.shape, y_test.shape)

  is_classification = True
  num_classes = 6

  x_train = pd.DataFrame(x_train)
  x_test = pd.DataFrame(x_test)
  y_train = pd.DataFrame(y_train - 1, dtype=np.int32).iloc[:, 0]
  y_test = pd.DataFrame(y_test - 1, dtype=np.int32).iloc[:, 0]

  return (x_train, y_train), (x_test, y_test)


def load_dataset(dataset):
    if dataset == "MNIST":
        return load_mnist()
    elif dataset == "MNIST-Fashion":
        return load_fashion()
    if dataset == "MICE":
        return load_mice()
    elif dataset == "COIL":
        return load_coil()
    elif dataset == "ISOLET":
        return load_isolet()
    elif dataset == "Activity":
        print(f"loading activity")
        return load_activity()
    else:
        print("Please specify a valid dataset")
        return None


##########################################
# added this section for data generation #
##########################################
def generate_data(n: int, 
                  p: int, 
                  k: int, 
                  response_type: str = 'continuous', 
                  rho: float = 0.5, 
                  seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate simulated data
    
    :param int n: sample size 
    :param int p: number of features 
    :param int k: number of non-zero coefficients
    :param string response_type: either continuous or categorical 
    :param float rho: correlation coefficient
    :param seed: random seed
    
    :return X: np.ndarray containing features 
    :return y: np.ndarray containing response 
    :return beta: np.ndarray containing true coefficient vector
    
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    if seed is not None:
        np.random.seed(seed)
    # generate correlation matrix and X
    ind = np.arange(p)
    dis = np.abs(ind[:, None] - ind)
    cov = rho ** dis 
    X = np.random.multivariate_normal(np.zeros(p), cov, size=n)
    
    # generate sparse coeff. it should be k non-zero entry 
    beta = np.zeros(p)
    nonzero_indices = np.random.choice(p, k, replace=False)
    beta[nonzero_indices] = np.random.uniform(-2, 2, k)
    
    # generate response
    if response_type == 'continuous':
        epsilon = np.random.normal(0, 1, n)
        y = X @ beta + epsilon
    else:  
        # binary case
        q = 1 / (1 + np.exp(-X @ beta))
        y = np.random.binomial(1, q)
    
    return X, y, beta


def load_parkinsons_data():
    """
    Load the Oxford Parkinson's Disease Detection dataset from OpenML
    Returns features (X) and target variable (y)
    in this dataset, the response vaariable is bianry. 
    """
    print("Loading Parkinson's dataset...")
    data = fetch_openml(data_id=1488, as_frame=True)
    X = data.data
    y = data.target.astype(float).to_numpy()  # Convert target to float
    
    print(f"Dataset loaded: {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Dataset loaded: X.shape is {X.shape} and y.shape is {y.shape}")
    return X, y

def load_california_housing_data(test=False):
    """
    Load the California Housing dataset from OpenML
    Returns features (X) and target variable (y)
    in this dataset, the response vaariable is continuous. 
    """
    print("Loading California Housing's dataset...")
    data = fetch_openml(data_id=537, as_frame=True)
    X = data.data
    y = data.target.astype(float).to_numpy()  # Convert target to float
    if test == True: 
        X = X[:100, ]
        y = y[:100]
    print(f"Dataset loaded: {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Dataset loaded: X.shape is {X.shape} and y.shape is {y.shape}")
    print(f"X is {X} and y is {y}")
    return X, y

def load_boston_housing_data(test=False):
    """
    Load the Boston Housing dataset from OpenML
    Returns features (X) and target variable (y)
    in this dataset, the response vaariable is continuous. 
    """
    print("Loading Boston Housing's dataset...")
    data = fetch_openml(data_id=43465, as_frame=True)
    X = data.data
    y = data.target.astype(float).to_numpy()  # Convert target to float
    if test == True: 
        X = X[:100, ]
        y = y[:100]
    print(f"Dataset loaded: {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Dataset loaded: X.shape is {X.shape} and y.shape is {y.shape}")
    print(f"X is {X} and y is {y}")
    return X, y
