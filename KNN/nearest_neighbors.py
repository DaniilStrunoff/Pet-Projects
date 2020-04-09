import numpy as np
import sklearn
from sklearn import datasets 
from sklearn.neighbors import NearestNeighbors
from distances import euclidean_distance, cosine_distance
EPS = 1e-5


class KNNClassifier:
    def __init__(self, k, strategy, metric, weights, test_block_size):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size
    
    def fit(self, X_train, y_train):
        self.y_train = y_train
        if self.strategy=="brute" or self.strategy=="ball_tree" or self.strategy=="kd_tree":
            model = NearestNeighbors(self.k, algorithm = self.strategy, metric = self.metric)
            self.nbrs = model.fit(X_train, y_train)
        else:
            self.X_train = X_train
            return
        return
    
    def find_kneighbors(self, X_test, return_distance):
        if self.strategy=="brute" or self.strategy=="ball_tree" or self.strategy=="kd_tree":
            distances, indeces = self.nbrs.kneighbors(X_test)
        else:
            if self.metric == 'euclidean':
                distances = euclidean_distance(X_test, self.X_train)
            elif self.metric == 'cosine':
                distances = cosine_distance(X_test, self.X_train)
            else: 
                print("Wrong metric!")
                return
            indeces = np.argpartition(distances, self.k, axis = 1)
            distances = np.take_along_axis(distances, indeces, axis = 1)
            distances = np.delete(distances, np.s_[self.k:], axis=1)
            dist_index = np.delete(indeces, np.s_[self.k:], axis=1)
            ind = np.argsort(distances, axis=1)
            self.distances = np.take_along_axis(distances, ind, axis=1)
            self.indeces = np.take_along_axis(indeces, ind, axis=1)
            return self.distances, self.indeces
        if(return_distance):
            return distances, indeces
        else:
            return indeces
    
    def predict(self, X_test):
        #_, self.indeces = self.nbrs.kneighbors(X_test)
        if(self.weights == False):
            self.indeces = self.find_kneighbors(X_test, False)
        else:
            self.distances, self.indeces = self.find_kneighbors(X_test, True)
        y_t = []
        C = [0]*10
        for i in range(X_test.shape[0]):
            C = [0]*10
            for j in range(self.k):
                if(self.weights):
                    C[int(self.y_train[self.indeces[i][j]])] += (1/(self.distances[i][j]+EPS))
                else:
                    C[int(self.y_train[self.indeces[i][j]])] += 1
            y_t.append(np.where(C == np.amax(C))[0][0])
        y_t = np.array(y_t)
        return y_t

def euclidean_distance(A, B):
    A2 = np.array([np.sum(A * A, axis=1)])
    B2 = np.array([np.sum(B * B, axis=1)])
    AB = np.dot(A,B.T)
    D = A2.T - 2 * AB + B2
    return np.sqrt(D)
def cosine_distance(A, B):
    A2vec = np.array(np.linalg.norm(A, axis=1))
    B2vec = np.array(np.linalg.norm(B, axis=1))
    AB = np.dot(A, B.T)
    cos_dist = AB / np.dot(A2vec[:,np.newaxis],B2vec[np.newaxis,:])
    return cos_dist