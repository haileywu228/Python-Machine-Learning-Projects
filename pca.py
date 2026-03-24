#!/usr/bin/env python
# coding: utf-8

# In[13]:


"""STA314 Homework 4.

Copyright and Usage Information
===============================

This file is provided solely for the personal and private use of students
taking STA314 at the University of Toronto St. George campus. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited.
"""


import matplotlib.pyplot as plt
import scipy.linalg as lin
import numpy as np
import requests
import io
import sklearn

from scipy.linalg import eigh
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[14]:


# You may find these helper functions useful

def sigmoid(x):
    """ Computes the element wise logistic sigmoid of x.
    """
    return 1.0 / (1.0 + np.exp(-x))


def load_train():
    """ Loads training data for digits_train.
    """
    response = requests.get("https://www.cs.toronto.edu/~cmaddis/courses/sta314_f25/data/digits.npz")
    response.raise_for_status()
    data = np.load(io.BytesIO(response.content))
    train_inputs = np.hstack((data["train2"], data["train3"]))
    train_targets = np.hstack((np.zeros((1, data["train2"].shape[1])), np.ones((1, data["train3"].shape[1]))))
    return train_inputs.T, train_targets.T


def load_train_small():
    """ Loads training data for digits_train_small.
    """
    response = requests.get("https://www.cs.toronto.edu/~cmaddis/courses/sta314_f25/data/digits.npz")
    response.raise_for_status()
    data = np.load(io.BytesIO(response.content))
    train_inputs = np.hstack((data["train2"][:, :2], data["train3"][:, :2]))
    train_targets = np.hstack((np.zeros((1, 2)), np.ones((1, 2))))
    return train_inputs.T, train_targets.T


def load_valid():
    """ Loads validation data.
    """
    response = requests.get("https://www.cs.toronto.edu/~cmaddis/courses/sta314_f25/data/digits.npz")
    response.raise_for_status()
    data = np.load(io.BytesIO(response.content))
    valid_inputs = np.hstack((data["valid2"], data["valid3"]))
    valid_targets = np.hstack((np.zeros((1, data["valid2"].shape[1])), np.ones((1, data["valid3"].shape[1]))))
    return valid_inputs.T, valid_targets.T


def load_test():
    """ Loads validation data.
    """
    response = requests.get("https://www.cs.toronto.edu/~cmaddis/courses/sta314_f25/data/digits.npz")
    response.raise_for_status()
    data = np.load(io.BytesIO(response.content))
    test_inputs = np.hstack((data["test2"], data["test3"]))
    test_targets = np.hstack((np.zeros((1, data["test2"].shape[1])), np.ones((1, data["test3"].shape[1]))))
    return test_inputs.T, test_targets.T


def plot_digits(digit_array):
    """ Visualizes each example in digit_array.
    :param digit_array: N x D array of pixel intensities.
    :return: None
    """
    CLASS_EXAMPLES_PER_PANE = 5

    # assume two evenly split classes
    examples_per_class = int(digit_array.shape[0] / 2)
    num_panes = int(np.ceil(float(examples_per_class) / CLASS_EXAMPLES_PER_PANE))

    for pane in range(num_panes):
        print("Displaying pane {}/{}".format(pane + 1, num_panes))

        top_start = pane * CLASS_EXAMPLES_PER_PANE
        top_end = min((pane + 1) * CLASS_EXAMPLES_PER_PANE, examples_per_class)
        top_pane_digits = extract_digits(digit_array, top_start, top_end)

        bottom_start = top_start + examples_per_class
        bottom_end = top_end + examples_per_class
        bottom_pane_digits = extract_digits(digit_array, bottom_start, bottom_end)

        show_pane(top_pane_digits, bottom_pane_digits)


def extract_digits(digit_array, start_index, end_index):
    """ Returns a list of 16 x 16 pixel intensity arrays starting
    at start_index and ending at end_index.
    """
    digits = []
    for index in range(start_index, end_index):
        digits.append(extract_digit_pixels(digit_array, index))
    return digits


def extract_digit_pixels(digit_array, index):
    """ Extracts the 16 x 16 pixel intensity array at the specified index.
    """
    return digit_array[index].reshape(16, 16).T


def show_pane(top_digits, bottom_digits):
    """ Displays two rows of digits on the screen.
    """
    all_digits = top_digits + bottom_digits
    fig, axes = plt.subplots(nrows=2, ncols=int(len(all_digits) / 2))
    for axis, digit in zip(axes.reshape(-1), all_digits):
        axis.imshow(digit, interpolation="nearest", cmap=plt.gray())
        axis.set_xticklabels([])
        axis.set_yticklabels([])
        axis.axis("off")
    # fig.subplots_adjust(wspace=0,
    #                     hspace=0)
    plt.tight_layout(h_pad=-7)
    plt.show()


def save_images(images, filename):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    plot_digits(images)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.savefig(filename)


# In[15]:


# You may also find this additional helper function useful

def show_eigenvectors(v):
    """ Display the eigenvectors as images.

    Arguments
        v: A matrix of dimension D x k that stores top k eigenvectors
    """
    plt.figure(1)
    plt.clf()
    for i in range(v.shape[1]):
        plt.subplot(1, v.shape[1], i + 1)
        plt.imshow(v[:,i].reshape(16, 16).T, cmap=plt.cm.gray)
    plt.show()


# In[16]:


def pca(x, k):
    """ PCA algorithm. Given the data matrix x and k,
    return the eigenvectors, mean of x, and the projected data (code vectors).

    Hint: You may use NumPy or SciPy to compute the eigenvectors/eigenvalues.

    Arguments
        x: A matrix with dimension N x D, where each row corresponds to one data point.
        k: int representing the number of dimension to reduce to.

    Returns
        v: A matrix of dimension D x k that stores top k eigenvectors
        mean: A vector of dimension D that represents the mean of x.
        proj_x: A matrix of dimension N x k where x is projected down to k dimension.
    """
    n, d = x.shape

    # Compute eigenvectors

    # compute the mean-centered data matrix
    x_cen = x - x.mean(axis=0)

    # compute covariance matrix
    x_cov = np.dot(x_cen.T, x_cen)/ (n - 1)

    # compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(x_cov)

    # sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # select the top k eigenvectors
    v = eigenvectors[:, :k]


    # Compute mean
    mean = np.mean(x, axis=0)


    # Project x onto v
    proj_x = np.dot(x_cen, v)


    return v, mean, proj_x


# In[30]:


def pca_classify():
    # Load all necessary datasets:
    x_train, y_train = load_train()
    x_valid, y_valid = load_valid()
    x_test, y_test = load_test()

    # Make sure the PCA algorithm is correctly implemented.
    v, mean, proj_x = pca(x_train, 5)

    # The below code visualize the eigenvectors.
    show_eigenvectors(v)


    k_lst = [2, 5, 10, 20, 30]

    # initiate an array to store the accuracy data
    val_accuracy = np.zeros(len(k_lst))
    test_accuracy = np.zeros(len(k_lst))

    # reshape y to a flat list of labels
    y_train = y_train.reshape(-1)
    y_valid = y_valid.reshape(-1)
    y_test = y_test.reshape(-1)

    for j, k in enumerate(k_lst):

        # compute the PCA-transformed data
        v_train, mean_train, proj_train = pca(x_train, k)
        v_valid, mean_valid, proj_valid = pca(x_valid, k)
        v_test, mean_test, proj_test = pca(x_test, k)

        # fit 1-NN classifier on the training code vector
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(proj_train, y_train)

        # predict label for the validation code vector
        y_pred_val = knn.predict(proj_valid)

        # predict label for the testing code vector
        y_pred_test = knn.predict(proj_test)

        # compute the accuracy and store the value
        acc_val = accuracy_score(y_valid, y_pred_val)
        acc_test = accuracy_score(y_test, y_pred_test)
        val_accuracy[j] = acc_val
        test_accuracy[j] = acc_test

    # plot validation accuracy vs. number of principal components k
    plt.plot(k_lst, val_accuracy)
    plt.title("Validation Accuracy of 1-NN on top PCs")
    plt.xlabel("Number of principal components")
    plt.ylabel("Validation accuracy")
    plt.show()
    print("Validation accuracies:", val_accuracy)
    print("Testing accuracies:", test_accuracy)


# In[31]:


pca_classify()

