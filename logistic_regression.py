## Logistic Regression Classifier 
# This project builds a logistic regression classifier from scratch to distinguish between
# handwritten digits 2 and 3. The model uses the sigmoid function to output probabilities
# and is trained using gradient descent with L2 regularization to prevent overfitting.
# Performance is evaluated using cross-entropy loss and accuracy on training, validation,
# and test datasets, with visualizations to analyze learning behavior.

# The dataset used in this project consists of a subset of handwritten digit images
# containing only digits 2 and 3. Each image is represented as a 16 × 16 grayscale
# pixel array, where pixel intensities are normalized to values between 0 and 1 and
# stored as flattened vectors of length 256. 
# The training dataset contains 300 samples per class, providing a balanced set of
# labeled examples for model estimation, while a smaller training subset with 2
# samples per class is available for testing and debugging. In addition, separate
# validation and test datasets are provided for model selection and final
# performance evaluation, ensuring that model performance is assessed on unseen
# data.

import matplotlib.pyplot as plt
import numpy as np
import requests
import io


# Helper Functions

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


# Main functions for Logistic Regression

def logistic_predict(weights, data):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          D is the number of features per example

    :param weights: A vector of weights with dimension (D + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x D, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """

    # Define parameters
    N = data.shape[0]

    # Add bias column of ones to data matrix
    X = np.hstack((data, np.ones((N, 1))))

    # Compute dot product Xw
    z = np.dot(X, weights)

    # Apply logistic function to get probabilities
    y = sigmoid(z)

    return y


def evaluate(targets, y):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          D is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """

    # Define parameters
    t = targets
    N = t.shape[0]

    # Compute averaged cross entropy
    ce = -(1 / N) * (np.dot(t.T, np.log(y)) + np.dot((1 - t).T, np.log(1 - y)))
    ce = ce.item()

    # Compute fraction of inputs classified correctly
    y_pred = (y >= 0.5).astype(int)
    frac_correct = float((y_pred == targets).mean())       

    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost of penalized logistic regression and its derivatives
    with respect to weights. Also return the predictions.

    Note: N is the number of examples
          D is the number of features per example

    :param weights: A vector of weights with dimension (D + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x D, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points, plus a penalty term.
           This is the objective that we want to minimize.
        df: (D+1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """

    # Define parameters
    lambd = hyperparameters["weight_regularization"]
    N = data.shape[0]
    w = weights

    # Add bias column to the design matrix
    X = np.hstack((data, np.ones((N, 1))))

    # Split weights to exclude bias
    w_nb = w[:-1]

    # Compute prediction y
    y = logistic_predict(w, data)

    # Compute average cross entropy
    ce, _ = evaluate(targets, y) 

    # Compute regularization term 
    reg = 0.5 * lambd * np.dot(w_nb.T, w_nb)

    # Average train loss f
    f = ce + reg

    # Compute gradient of cross-entropy loss
    grad_loss = np.dot(X.T, (y - targets)) / N

    # Compute gradient of the regularization term 
    grad_reg = np.vstack((lambd * w_nb, [[0.0]]))

    # Total gradient
    df = grad_loss + grad_reg

    return f, df, y


def run_logistic_regression():
    x_train, y_train = load_train_small()
    x_valid, y_valid = load_valid()
    x_test, y_test = load_test()

    # Define parameters
    N, d = x_train.shape

    # Set the hyperparameters for the learning rate, the number of iterations                                                     #
    hyperparameters = {
        "learning_rate": 0.5,
        "weight_regularization": 0.01,
        "num_iterations": 100
    }

    # Begin learning with gradient descent                                     

    # Define parameters
    learning_rate = hyperparameters["learning_rate"]
    num_iteration = hyperparameters["num_iterations"]
    lambd = hyperparameters["weight_regularization"]

    # Perform gradient descent 
    w = np.zeros((d + 1, 1))
    train_ce_track = []
    valid_ce_track = []

    for n in range(num_iteration):
        _, df, _ = logistic(w, x_train, y_train, hyperparameters)
        w = w - learning_rate * df

        y_train_p = logistic_predict(w, x_train)
        ce_train, _ = evaluate(y_train, y_train_p)
        train_ce_track.append(float(ce_train))

        y_val_p = logistic_predict(w, x_valid)
        ce_valid, _ = evaluate(y_valid, y_val_p)
        valid_ce_track.append(float(ce_valid)) 

    # Compute the cross-entropy loss and classification accuracy on validation set 
    y_val_p = logistic_predict(w, x_valid)
    ce_val, acc_val = evaluate(y_valid, y_val_p)

    # Print the results
    # print(f"Learning Rate={learning_rate}, Iteration={num_iteration}")
    # print(f"Validation: CE={ce_val:.4f}, Accuracy={acc_val:.4f}")


    # The following hyperparameters were experimented
    # learning_rate = [0.01, 0.1, 0.5, 2.0]
    # num_iteration = [50, 100, 500, 1000]

    # Experiment shows that learning rate = 2.0 and num_iteration = 50 produced the 
    # lowest cross entropy loss and relatively high accuracy. They are then chosen 
    # as the hyperparameters for the final model. 


    # Testing 
    # Compute the cross-entropy loss and classification accuracy on test set
    y_test_p = logistic_predict(w, x_test)
    ce_test, acc_test = evaluate(y_test, y_test_p)

    # Compute the cross-entropy loss and classification accuracy on train set
    # Training results
    y_train_p = logistic_predict(w, x_train)
    ce_train, acc_train = evaluate(y_train, y_train_p)

    # Print all results
    print(f"Learning Rate={learning_rate},  Iteration={num_iteration},  Weight Regularization = {lambd}")
    print(f"Training: CE={ce_train:.4f}, Accuracy={acc_train:.4f}")
    print(f"Validation: CE={ce_val:.4f}, Accuracy={acc_val:.4f}")
    print(f"Testing: CE={ce_test:.4f}, Accuracy={acc_test:.4f}")


    # Plot CE vs. iteration for training & validation set

    # Helper function to plot CE vs. iteration
    def plot_ce_curves(train_ce_track, valid_ce_track, learning_rate):
        """
        Plots cross-entropy loss vs. iteration for training and validation sets.
        Parameters:
        - train_ce_track: list of CE values from training set
        - valid_ce_track: list of CE values from validation set
        - learning_rate: float, used in plot title
        """
        plt.figure(figsize=(8, 5))
        plt.plot(train_ce_track, label='Training CE', color='blue', linewidth=2)
        plt.plot(valid_ce_track, label='Validation CE', color='orange', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Cross-Entropy Loss')
        plt.title(f'Cross-Entropy vs. Iteration (Learning Rate = {learning_rate})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Plot CE vs. iteration
    plot_ce_curves(train_ce_track, valid_ce_track, learning_rate)

run_logistic_regression()

# Helper function to plot CE vs. weight regulation
def plot_validation_ce_vs_lambda(lambda_values, val_ce_results, dataset_name="Full Training Set"):
    """
    Plots validation cross-entropy vs. weight regularization λ.
    Parameters:
    - lambda_values: list of λ values
    - val_ce_results: list of corresponding validation CE values
    - dataset_name: string to label the plot
    """
    plt.figure(figsize=(8, 5))
    plt.plot(lambda_values, val_ce_results, marker='o', linestyle='-', color='purple')
    plt.xlabel("Weight Regularization λ")
    plt.ylabel("Validation Cross-Entropy")
    plt.title(f"Validation CE vs. λ ({dataset_name})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Record validation CE with weight regularization

# The lambda values of 0, 0.001, 0.01, 0.1, 1.0 are maunally set as the parameters in the model 
# and the validation CE are recorded in valida_ce_w. 

# Full training set 
lambda_values = [0, 0.001, 0.01, 0.1, 1.0]
val_ce_results_f = [0.0686, 0.0702, 0.0845, 0.1764, 1.1408]
val_ce_results_s = [0.4959, 0.4950, 0.4926, 0.5139, 1.2095]


# Plot validation CE vs. weight regulation
# Full training set 
plot_validation_ce_vs_lambda(lambda_values, val_ce_results_f, dataset_name="Full Training Set")
plot_validation_ce_vs_lambda(lambda_values, val_ce_results_s, dataset_name="Small Training Set")
