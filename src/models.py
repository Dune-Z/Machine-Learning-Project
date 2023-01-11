import numpy as np
import time
import matplotlib.pyplot as plt
import tqdm
import pandas as pd

from scipy.optimize import minimize


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


# Define the multiclass logistic regression function to be optimized
def multiclass_logreg(w, X, y, num_classes, alpha):
    m, n = X.shape
    w = w.reshape((num_classes, n))
    logits = X @ w.T
    prob = softmax(logits)
    ll = -np.mean(np.log(prob[range(m), y])) + 0.5 * alpha * np.sum(w ** 2)
    return ll


# Define the gradient of the multiclass logistic regression function
def grad_multiclass_logreg(w, X, y, num_classes, alpha):
    m, n = X.shape
    w = w.reshape((num_classes, n))
    logits = X @ w.T
    prob = softmax(logits)
    prob[range(m), y] -= 1
    grad = prob.T @ X / m + alpha * w
    return grad.ravel()


class LogisticClassifier():

    def __init__(self, num_class=3, alpha=0.1) -> None:
        self.w = None
        self.d = None
        self.num_class = num_class
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray, method='BFGS'):
        self.d = X.shape[1]
        initial_w = np.random.randn(self.num_class * self.d)
        res = minimize(multiclass_logreg,
                       initial_w,
                       args=(X, y, self.num_class, self.alpha),
                       jac=grad_multiclass_logreg,
                       tol=1e-4,
                       method=method)

        self.w = res.x.reshape((self.num_class, self.d))

    def predict_proba(self, X: np.ndarray):
        """
        Predict probability of each class
        """
        logits = X @ self.w.T
        return softmax(logits)

    def predict(self, X: np.ndarray):
        """
        Predict the labels.
        """
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)


"""
Previous version
"""


class RandomClassifier:
    """
    Generate a random number between 0 and 2 for prediction.
    For testing purpose.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.random.randint(0, 3, size=X.shape[0])


def softmax_(X):
    """
    Softmax function.
    """
    X_max = np.max(X, axis=1)
    # Broadcast manually
    X_max = np.stack((X_max, X_max, X_max)).T
    X = X - X_max
    X_exp = np.exp(X)
    X_sum = np.sum(X_exp, axis=1)
    Z = X_exp.T / X_sum
    return Z.T


def log(x):
    """
    Log function.
    """
    if x == 0:
        return -1e10
    else:
        return np.log(x)


def sigmoid(x):
    '''
    Sigmoid function.
    '''
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)


sigmoid = np.vectorize(sigmoid)
log = np.vectorize(log)


class LogisticClassifier_pre:
    """
    Multinomial Logistic Regression using GD with L2 regularization.
    """

    def __init__(self,
                 alpha=0.01,
                 learning_rate=1e-5,
                 target_accuracy=0.90,
                 max_iter=500,
                 random_state=0):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.target_accuracy = target_accuracy
        self.max_iter = max_iter
        self.random_state = random_state

        self.n = None  # Number of samples
        self.d = None  # Number of features
        self.X = None  # Feature matrix with bias
        self.y = None  # Label vector
        self.w = None  # Weight vector
        self.prob = None  # Predicted probabilities
        self.grad = None  # Gradient of loss
        self.loss_list = []  # Cross-entropy loss
        self.score_list = []  # Accuracy score
        self.total_iter = 0  # Total number of iterations
        self.total_time = 0  # Total time elapsed
        self.converged = False  # Whether the algorithm converged
        self.num_classes = None  # Number of classes
        self.Y_onehot = None

    def fit(self,
            X,
            y,
            w0=None,
            verbose=False,
            record_loss=False,
            record_score=True):
        """
        Train the classifier.
        """
        # Set the random seed
        np.random.seed(self.random_state)
        # Initialize the parameters
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.X = np.hstack((np.ones((self.n, 1)), X))
        self.y = y.astype(int)
        self.Y_onehot = pd.get_dummies(y)[[0, 1, 2]].values
        self.num_classes = 3
        self.w = w0
        if self.w is None:
            self.w = np.random.randn(self.d + 1, self.num_classes)
        self.score_list.clear()
        self.loss_list.clear()
        self.total_time = 0
        self.converged = False
        # Minimize the loss function
        for self.total_iter in tqdm.trange(1, self.max_iter + 1):
            # Start timer
            start = time.time()
            # Compute the gradient of the loss function
            self.grad = self.gradient()

            # Update the weights
            self.w -= self.learning_rate * self.grad
            # Stop timer
            duration = time.time() - start
            self.total_time += duration

            # Compute the cross-entropy loss
            #    loss = self.loss()
            #    if record_loss:
            #        self.loss_list.append(self.loss())
            # Compute the accuracy score
            score = self.score()
            if record_score:
                self.score_list.append(score)
            # Print the progress
            if verbose:
                print(f'Iteration {self.total_iter:4d}: '
                      #  f'Loss = {loss:.4f}, '
                      f'Score = {score:.4f}, '
                      f'Time = {duration:.4f} sec')
            # Stop if the target accuracy is reached
            if self.target_accuracy < score:
                self.converged = True
                break

    def predict(self, X):
        """
        Predict the labels.
        """
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1).astype(int)

    def predict_proba(self, X):
        """
        Predict the probabilities of the positive class.
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return softmax_(-X @ self.w)

    def loss(self):
        """
        Evaluate the cross-entropy loss on the training set.
        """
        Y = self.Y_onehot
        Z = -self.X @ self.w
        loss = 1 / self.n * (np.trace(self.X @ self.w @ Y.T) + np.sum(
            log(np.sum(np.exp(Z), axis=1)))) + self.alpha / 2 * np.sum(self.w**
                                                                       2)
        return loss

    def gradient(self):
        Z = -self.X @ self.w
        self.prob = softmax_(Z)
        Y = self.Y_onehot
        gd = 1 / self.n * (self.X.T @ (Y - self.prob)) + self.alpha * self.w
        return gd

    def score(self):
        """
        Evaluate the accuracy on the training set.
        """
        return np.mean(np.argmax(self.prob, axis=1).astype(int) == self.y)


class BinaryClassifier:
    '''
    Binomial Logistic Regression using GD, SGD, or mini-batch SGD with L2 regularization.
    '''

    def __init__(self,
                 alpha=0.01,
                 learning_rate=None,
                 batch_size=None,
                 target_accuracy=0.90,
                 max_iter=5000,
                 random_state=0):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_accuracy = target_accuracy
        self.max_iter = max_iter
        self.random_state = random_state

        self.n = None  # Number of samples
        self.d = None  # Number of features
        self.X = None  # Feature matrix with bias
        self.y = None  # Label vector
        self.w = None  # Weight vector
        self.L = None  # Lipschitz constant
        self.prob = None  # Predicted probabilities
        self.grad = None  # Gradient of loss
        self.loss_list = []  # Cross-entropy loss
        self.score_list = []  # Accuracy score
        self.total_iter = 0  # Total number of iterations
        self.total_time = 0  # Total time elapsed
        self.converged = False  # Whether the algorithm converged

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            w0=None,
            verbose=False,
            record_loss=False,
            record_score=True):
        '''
        Train the classifier.
        '''
        # Set the random seed
        np.random.seed(self.random_state)
        # Initialize the parameters
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.X = np.hstack((np.ones((self.n, 1)), X))
        self.y = y
        self.w = w0.copy()
        if self.w is None:
            self.w = np.random.randn(self.d + 1)
        if self.learning_rate is None:
            # Estimate the Lipschitz constant (See Problem 1)
            self.L = (self.d + 1) / 4
        else:
            self.L = 1 / self.learning_rate
        self.score_list.clear()
        self.loss_list.clear()
        self.total_time = 0
        self.converged = False
        # Minimize the loss function
        for self.total_iter in range(1, self.max_iter + 1):
            # Start timer
            start = time.time()
            # Compute the gradient of the loss function
            if self.batch_size is None:
                # Full gradient
                self.prob = sigmoid(self.X @ self.w)
                self.grad = self.X.T @ (self.prob - self.y) / self.n
            else:
                # Stochastic gradient
                idx = np.random.randint(self.X.shape[0], size=self.batch_size)
                self.prob = sigmoid(self.X[idx] @ self.w)
                self.grad = self.X[idx].T @ (self.prob -
                                             self.y[idx]) / self.batch_size
            # Update the weights
            self.w -= 1 / (self.L + self.alpha) * (self.grad +
                                                   self.alpha * self.w)
            # Stop timer
            duration = time.time() - start
            self.total_time += duration
            # As we need to evaluate the accuracy score in each iteration, we must compute the full probability vector here, which cause the stochastic gradient descent to be as slow as the full gradient descent.
            if self.batch_size is not None:
                self.prob = sigmoid(self.X @ self.w)
            # Compute the cross-entropy loss
            loss = self.loss()
            if record_loss:
                self.loss_list.append(self.loss())
            # Compute the accuracy score
            score = self.score()
            if record_score:
                self.score_list.append(score)
            # Print the progress
            if verbose:
                print(f'Iteration {self.total_iter:4d}: '
                      f'Loss = {loss:.4f}, '
                      f'Score = {score:.4f}, '
                      f'Time = {duration:.4f} sec')
            # Stop if the target accuracy is reached
            if self.target_accuracy < score:
                self.converged = True
                break

    def predict(self, X):
        '''
        Predict the labels.
        '''
        prob = self.predict_proba(X)
        return (prob > 0.5).astype(int)

    def predict_proba(self, X):
        '''
        Predict the probabilities of the positive class.
        '''
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return sigmoid(X @ self.w)

    def loss(self):
        '''
        Evaluate the cross-entropy loss on the training set.
        '''
        return -np.mean(self.y * log(self.prob) +
                        (1 - self.y) * log(1 - self.prob)
                        ) + self.alpha / 2 * np.sum(self.w**2)

    def score(self):
        '''
        Evaluate the accuracy on the training set.
        '''
        return np.mean((self.prob >= 0.5).astype(int) == self.y)


class OrdinalClassifier:
    """
    Ordinal Classifier
    Using two Logistic Regression classifier to predict P(Target > Small) and P(Target > True to Size)
    Target should be encoded as 0, 1, 2
    """

    def __init__(self,
                 alpha=0.01,
                 learning_rate=1e-5,
                 target_accuracy=0.90,
                 max_iter=100,
                 random_state=0):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.target_accuracy = target_accuracy
        self.max_iter = max_iter
        self.random_state = random_state

        self.clf_low = BinaryClassifier(max_iter=self.max_iter,
                                        alpha=self.alpha)
        self.clf_high = BinaryClassifier(max_iter=self.max_iter,
                                         alpha=self.alpha)

    def fit(self, X: np.ndarray, y: np.ndarray, w0_high=None, w0_low=None):

        y_low = (y != 0).astype(int)
        y_high = (y == 2).astype(int)

        if w0_high == None:
            w0_high = np.random.randn(X.shape[1] + 1)
        if w0_low == None:
            w0_low = np.random.randn(X.shape[1] + 1)

        self.clf_low.fit(X, y_low, w0=w0_low)
        self.clf_high.fit(X, y_high, w0=w0_high)

    def predict_proba(self, X):
        prob_0 = 1 - self.clf_low.predict_proba(X)
        prob_1 = self.clf_low.predict_proba(X) - self.clf_high.predict_proba(X)
        prob_2 = self.clf_high.predict_proba(X)

        prob = np.stack([prob_0, prob_1, prob_2]).T
        return prob

    def predict(self, X):
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1) + 1

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
