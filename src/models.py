import numpy as np
import time
import matplotlib.pyplot as plt
import tqdm

from utils import onehot_encoding

def softmax(X):
    '''
    Softmax function.
    '''
    X_max = np.max(X, axis=1)
    # Broadcast manually
    X_max = np.stack((X_max, X_max, X_max)).T
    X = X - X_max
    X_exp = np.exp(X)
    X_sum = np.sum(X_exp, axis=1)
    Z = X_exp.T / X_sum
    return Z.T


def log(x):
    '''
    Log function.
    '''
    if x == 0:
        return -1e10
    else:
        return np.log(x)

log = np.vectorize(log)

class LogisticClassifier:
    '''
    Multiclass Logistic Regression using GD with L2 regularization.

    存在的问题：\n
    1. loss计算十分缓慢 (且存在overflow bug)
    2. 模型训练时score波动较大
    '''

    def __init__(self,
                 alpha=0.01,
                 learning_rate=1e-5,
                 target_accuracy=0.90,
                 max_iter=10,
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
        self.num_classes = None # Number of classes
        self.Y_onehot = None

    def fit(self,
            X,
            y,
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
        self.Y_onehot = onehot_encoding(self.y, ['fit'])[[0,1,2]].values
        self.num_classes = len(y.unique())
        self.w = w0.copy()
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
        '''
        Predict the labels.
        '''
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1).astype(int)

    def predict_proba(self, X):
        '''
        Predict the probabilities of the positive class.
        '''
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return softmax(-X @ self.w)

    def loss(self):
        '''
        Evaluate the cross-entropy loss on the training set.
        '''
        Y = self.Y_onehot
        Z = -self.X @ self.w
        loss = 1/self.n * (np.trace(self.X @ self.w @ Y.T) + np.sum(log(np.sum(np.exp(Z), axis=1)))) + self.alpha / 2 * np.sum(self.w**2)
        return loss

    def gradient(self):
        Z = -self.X @ self.w
        self.prob = softmax(Z)
        Y = self.Y_onehot
        gd = 1/self.n * (self.X.T @ (Y - self.prob)) + self.alpha * self.w
        return gd

    def score(self):
        '''
        Evaluate the accuracy on the training set.
        '''
        return np.mean(np.argmax(self.prob, axis=1).astype(int) == self.y)

    def plot_loss(self, ax=None, **kwargs):
        '''
        Plot the loss function.
        '''
        if ax is None:
            ax = plt.gca()
        ax.plot(self.loss_list, **kwargs)

    def plot_score(self, ax=None, **kwargs):
        '''
        Plot the accuracy score.
        '''
        if ax is None:
            ax = plt.gca()
        ax.plot(self.score_list, **kwargs)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots(1, 2, figsize=(12, 4))
        self.plot_loss(ax[0], **kwargs)
        self.plot_score(ax[1], **kwargs)
