"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

This solution will run but the dimensionality reduction and
the classifier are not doing anything useful, so it will
produce a very poor result.

version: v1.0
"""
from typing import List

import numpy as np
import scipy.linalg


N_DIMENSIONS = 10

def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
    """Classify a set of feature vectors using a training set.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # k nearest neighbour implementation
    x = np.dot(test, train.transpose()) # calculates dot product
    modtest = np.sqrt(np.sum(test * test, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose())  # cosine distance
    nearest_four = np.argsort(dist)[:,-4:] # select 4 greatest values
    flattened_nearest = nearest_four.flatten() # all the args we want from train_labels
    nearest_labels = train_labels[flattened_nearest] # 4 nearest labels for every square

    # convert to ascii so bincount can operate
    ascii_labels = np.array([[ord(char) for char in row] for row in nearest_labels])

    four_labels = np.reshape(ascii_labels,(1600,4))

    # this is chosing what to classify each square as by selecting most common neighbour
    ascii_nearest = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),axis=1,
        arr=four_labels)

    # convert ascii back to the label
    labels_chosen = np.array([chr(value) for value in ascii_nearest])
    return labels_chosen

def divergence(class1, class2):
    """compute a vector of 1-D divergences
    
    class1 - data matrix for class 1, each row is a sample
    class2 - data matrix for class 2
    
    returns: d12 - a vector of 1-D divergence scores
    """

    # Compute the mean and variance of each feature vector element
    m1 = np.mean(class1, axis=0)
    m2 = np.mean(class2, axis=0)
    v1 = np.var(class1, axis=0)
    v2 = np.var(class2, axis=0)

    d12 = 0.5*(v1/v2 + v2/v1 - 2) + 0.5*(m1-m2)*(m1-m2)*(1.0 / v1 + 1.0 / v2)

    return d12

def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    The feature vectors are stored in the rows of 2-D array data, (i.e., a data matrix).
    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
        
    if data.size == 16000000: # we are training
        # compute the first 40 principle components (eigenvectors)
        # the irrelevant noise component will largely be removed
        covx = np.cov(data, rowvar=0)
        N = covx.shape[0]
        w, v = scipy.linalg.eigh(covx, eigvals=(N - 40, N - 1))
        v = np.fliplr(v)
        list_v = v.tolist()
        model['v'] = list_v  # store v in the model

        # centre the data by subtracting the mean data vector (before transforming it)
        pca_data = np.dot((data - np.mean(data)), v)

        train_data = model.get('labels_train')
        train_labels = np.array(train_data)

        possible_squares = list(set(model['labels_train'])) # list of all the square states
        d = 0
        # find the summed divergence of every square state combination
        for square_state in possible_squares:
            index = possible_squares.index(square_state)
            state_data = pca_data[train_labels[:] == square_state, :]
            for diff_state in possible_squares[index+1:]:
                diff_state_data = pca_data[train_labels[:] == diff_state, :]
                d12 = divergence(state_data, diff_state_data)
                d = d + d12
        # select the 10 features with the greatest divergence 
        sorted_indexes = np.argsort(-d)
        features = sorted_indexes[0:N_DIMENSIONS]
        features_list = features.tolist()
        model['features'] = features_list # store the features in the model
    else: # we are evaluating
        v_from_model = model.get('v') # extract v from model
        train_v = np.array(v_from_model)
        pca_data = np.dot(data - np.mean(data), train_v)

    features_from_model = model.get('features') 
    train_features = np.array(features_from_model)
    reduced_data = pca_data[:, train_features]
    return reduced_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Note, the contents of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """
    model = {}
    model["labels_train"] = labels_train.tolist() 
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model) 
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """
    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        fvectors[i, :] = image.reshape(1, n_features)

    return fvectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Get some data out of the model
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    labels = classify(fvectors_train, labels_train, fvectors_test)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    """
        The following code adds coordinates (x,y) of each square onto the end of
        fvectors_train, which has been cut down to 8 features to accommodate for the
        last 2 becomming board position. It produced scores:
          - clean board score : 91.6%
          - noisy board score : 82%

        Later I tried just having a square number (1-64) with 9 other features, and this
        produced a better score, so that is the code I have used.
        
        first_8_fvectors = fvectors_train[:, :8]

        x = 1
        y = 1
        boards_fvectors_train = np.empty(shape=[0, 10])

        # adds the coordinates of each square to the end of its features
        for row in first_8_fvectors:
            if x > 8: # end of the row
                x = 1
                y += 1
            if y > 8: # end of the board
                y = 1
            row = np.insert(row,8,[x,y])
            x += 1
            boards_fvectors_train = np.append(boards_fvectors_train, [row], axis=0)
"""
    first_9_fvectors = fvectors_train[:, :9]

    square = 1
    boards_fvectors_train = np.empty(shape=[0, 10])

    # adds the coordinates of each square to the end of its features
    for row in first_9_fvectors:
        if square > 64:
            square = 1
        row = np.insert(row,9,[square])
        square += 1
        boards_fvectors_train = np.append(boards_fvectors_train, [row], axis=0)
    
    boards_labels = classify(boards_fvectors_train, labels_train, fvectors_test)
    return boards_labels
