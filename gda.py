#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


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


# In[4]:


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(t^(i) | x^(i)) )

    i.e. the average log likelihood that the model assigns to the correct class label.

    Arguments
        digits: size N x 256 numpy array with the images
        labels: size N numpy array with the labels
        means: size 2 x 256 numpy array with the 2 class means
        covariances: size 2 x 256 x 256 numpy array with the 2 class covariances

    Returns
        average conditional log-likelihood.
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    assert len(digits) == len(labels)
    sample_size = len(digits)
    total_prob = 0
    for j in range(sample_size):
        label = int(labels[j])
        total_prob += cond_likelihood[j][label]
    return total_prob/sample_size


# Finish the five functions below.

# In[5]:


def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class. You may iterate over
    the possible label values (0 or 1 corresponding to digits "2" or "3"), 
    but otherwise make sure that your code is vectorized.

    Arguments
        train_data: size N x 256 numpy array with the images
        train_labels: size N numpy array with corresponding labels

    Returns
        means: size 2 x 256 numpy array with the ith row corresponding
               to the mean estimate for digit class i
    '''
    # Initialize array to store means
    means = np.zeros((2, 256))

    for k in [0, 1]:

        # filter rows by class
        class_data = train_data[train_labels == k] 

        # mean across rows
        means[k] = class_data.mean(axis=0)

    return means


def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class. You may iterate over
    the possible label values (0 or 1 corresponding to digits "2" or "3"), 
    but otherwise make sure that your code is vectorized.

    Arguments
        train_data: size N x 256 numpy array with the images
        train_labels: size N numpy array with corresponding labels

    Returns
        covariances: size 2 x 256 x 256 numpy array with the ith row corresponding
               to the covariance matrix estimate for label i
    '''
    # Initialize array to store covariances
    covariances = np.zeros((2, 256, 256))

    for k in [0, 1]:
        class_data = train_data[train_labels == k]
        N = class_data.shape[0]

        # Compute mean vector for class k
        mean_k = class_data.mean(axis=0)

        # Center the data
        centered = class_data - mean_k

        # Compute covariance matrix
        cov_k = np.dot(centered.T, centered) / N

        # Add 0.1 * I for numerical stability
        cov_k += 0.1 * np.eye(256)

        covariances[k] = cov_k

    return covariances


def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood log p(x|t). You may iterate over
    the possible label values (0 or 1 corresponding to digits "2" or "3"), 
    but otherwise make sure that your code is vectorized.

    Arguments
        digits: size N x 256 numpy array with the images
        means: size 2 x 256 numpy array with the 2 class means
        covariances: size 2 x 256 x 256 numpy array with the 2 class covariances

    Returns
        likelihoods: size N x 2 numpy array with the ith row corresponding
               to logp(x^(i) | t) for t in {0, 1}
    '''
    N = digits.shape[0]
    D = digits.shape[1]
    likelihoods = np.zeros((N, 2))

    for k in [0, 1]:
        mu_k = means[k]                      
        sigma_k = covariances[k]
        sigma_inv = np.linalg.inv(sigma_k)
        log_det = np.linalg.slogdet(sigma_k)[1]

        # Center the data
        centered = digits - mu_k

        # Mahalanobis term: (x - mu)^T Σ⁻¹ (x - mu)
        mahalanobis = np.sum(centered @ sigma_inv * centered, axis=1)

        # Full log-likelihood
        likelihoods[:, k] = -0.5 * (mahalanobis + log_det + D * np.log(2 * np.pi))

    return likelihoods



def conditional_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood log p(t|x). Make sure that your code
    is vectorized. Do not iterate over the label values explicitly in Python.

    Arguments
        digits: size N x 256 numpy array with the images
        means: size 2 x 256 numpy array with the 2 class means
        covariances: size 2 x 256 x 256 numpy array with the 2 class covariances

    Returns
        likelihoods: size N x 2 numpy array with the ith row corresponding
               to logp(t | x^(i)) for t in {0, 1}
    '''
    N = digits.shape[0]
    D = digits.shape[1]
    likelihoods = np.zeros((N, 2))

    # Compute prior
    log_prior = np.log(0.5)

    # Compute likelihood
    for k in [0, 1]:
        mu_k = means[k]
        sigma_k = covariances[k]
        sigma_inv = np.linalg.inv(sigma_k)
        log_det = np.linalg.slogdet(sigma_k)[1]

        centered = digits - mu_k
        mahalanobis = np.sum(centered @ sigma_inv * centered, axis=1)

        log_likelihood = -0.5 * (mahalanobis + log_det + D * np.log(2 * np.pi))
        likelihoods[:, k] = log_likelihood + log_prior

    return likelihoods



def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class. 
    Make sure that your code is vectorized. Do not iterate over 
    the label values explicitly in Python.

    Arguments
        digits: size N x 256 numpy array with the images
        means: size 2 x 256 numpy array with the 2 class means
        covariances: size 2 x 256 x 256 numpy array with the 2 class covariances

    Returns
        pred: size N numpy array with the ith element corresponding
               to argmax_t log p(t | x^(i))
    '''
    N = digits.shape[0]
    D = digits.shape[1]
    log_posteriors = np.zeros((N, 2))
    log_prior = np.log(0.5)

    # Compute log p(t | x) for each class

    for k in [0, 1]:
        mu_k = means[k]
        sigma_k = covariances[k]
        sigma_inv = np.linalg.inv(sigma_k)
        log_det = np.linalg.slogdet(sigma_k)[1]

        centered = digits - mu_k
        mahalanobis = np.sum(centered @ sigma_inv * centered, axis=1)

        log_likelihood = -0.5 * (mahalanobis + log_det + D * np.log(2 * np.pi))
        log_posteriors[:, k] = log_likelihood + log_prior

    # Classify by highest posterior
    pred = np.argmax(log_posteriors, axis=1)

    return pred


# In[6]:


def main():
    x_train, y_train = load_train()
    x_test, y_test = load_test()
    y_train, y_test = y_train.flatten(), y_test.flatten()

    # Fit the model
    means = compute_mean_mles(x_train, y_train)
    covariances = compute_sigma_mles(x_train, y_train)

    # Evaluation
    train_log_llh = avg_conditional_likelihood(x_train, y_train, means, covariances)
    test_log_llh = avg_conditional_likelihood(x_test, y_test, means, covariances)

    print('Train average conditional log-likelihood: ', train_log_llh)
    print('Test average conditional log-likelihood: ', test_log_llh)

    train_posterior_result = classify_data(x_train, means, covariances)
    test_posterior_result = classify_data(x_test, means, covariances)

    train_accuracy = np.mean(y_train.astype(int) == train_posterior_result)
    test_accuracy = np.mean(y_test.astype(int) == test_posterior_result)

    print('Train posterior accuracy: ', train_accuracy)
    print('Test posterior accuracy: ', test_accuracy)

    for i in range(2):
        (e_val, e_vec) = np.linalg.eigh(covariances[i])
        # In particular, note the axis to access the eigenvector
        curr_leading_evec = e_vec[:,np.argmax(e_val)].reshape((16,16))
        plt.subplot(3,4,i+1)
        plt.imshow(curr_leading_evec, cmap='gray')
    plt.show()


# In[7]:


main()

