import numpy as np
import math

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = make_classification()

X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], random_state=0)


class LogisticRegression:
    
    def __init__(self, alpha=0.5, treshold=0.5, random_state=None):
        self.alpha = alpha
        self.treshold = treshold
        self.random_state = random_state
        
    
    def fit(self, X_train, y_train):
        i = 0
        max_i = 250
        self.W = self.init_weights(X_train.shape[1])
        self.error = 10000
        
        while self.error > 0.01 and i < max_i:
            z = np.dot(X_train, self.W.T)
            a = self.logit(z)
            self.error = self.loss(X_train, y_train, self.W)
            self.W = self.W - self.alpha * self.gradient(X_train, y_train, self.W)
            
            i += 1
            
        
    def loss(self, X, y, w):
        log = np.vectorize(math.log)
        z = np.dot(X, w.T)
        return -np.sum(y*log(logit(z)) + (1-y)*(log(1-logit(z))))
    
    
    def predict(self, X_test):
        pred = np.dot(X_test, self.W.T)
        pred[pred >= self.treshold] = 1
        pred[pred < self.treshold] = 0
        return pred
    
    
    def predict_proba(self, X_test):
        return np.dot(X_test, self.W.T)
    
    
    def gradient(self, X, y, w):
        z = np.dot(X, w.T)
        a = logit(z)

        # da_ = dl/da = -y/a + (1-y)/(1-a)
        # dz_ = dl/da * da/dz = da_ * a(1-a) = a - y
        # dw_ = dl/da * da/dz * dz/dw = dz_ * X   -- gradient for 1 sample.

        gradient = []
        # calculating gradients for each sample and then averaging
        for i in range(X.shape[0]):
            gradient.append(X[i] * (a - y)[i])
        return np.mean(np.array(gradient), axis=0)
    
    
    def logit(self, z):
        exp = np.vectorize(math.exp)
        return exp(z)/(1 + exp(z))
    
    
    def init_weights(self, size):
        if self.random_state != None:
            np.random.seed(self.random_state)
        return np.random.random_sample(size)

lr = LogisticRegression()

lr.fit(X_train, y_train)
y_predict = lr.predict(X_test)
accuracy_score(y_test, y_predict), lr.error




