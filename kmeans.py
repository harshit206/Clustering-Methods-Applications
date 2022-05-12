import numpy as np
from numpy.linalg import norm


class Kmeans:
    def __init__(self,variant = None, max_iter=30, tolerance=1e-12):
        self.max_iter=max_iter
        self.variant = variant
        self.tolerance = tolerance
        
    def __calculate_centroid(self,X,Y,k):
        centroids = np.zeros(shape=(k,X.shape[1]))
        for i in range(k):
            centroids[i,:] = np.mean(X[Y==i],axis=0)
        return centroids

    def __kmeansplus(self,X,k):
        indices = np.random.choice(np.unique(X,axis=0).shape[0], size=1,replace=False)  
        centroids = X[indices]
        labels = np.zeros(shape=(X.shape[0]))


        for i in range(k-1):
            # print("k:",i)
            distance = np.zeros(shape=(X.shape[0],i+1))
            # print(centroids)
            for j in range(len(X)):
                distance[j,:] = norm(X[j,:]-centroids, axis=1)
                cluster_index = np.argmin(distance[j,:], axis=0)
                labels[j] = cluster_index

            cents = np.zeros(shape=X.shape)
            labels = labels.astype('int')
            cents = centroids[labels]
            max_distance = np.argmax(norm(X-cents,axis=1))
            centroids = list(centroids)
            centroids.append(X[max_distance])
            centroids = np.array(centroids)
        return centroids


    def fit(self, X, k):
        self.k = k
        self.X = X
        
        if self.variant=='kmeans++':
            print("Running KMeans++")
            centroids = self.__kmeansplus(X,k)
            
        else:
            indices = np.random.choice(np.unique(X,axis=0).shape[0], size=k,replace=False)
            centroids = X[indices]
        
        distance = np.zeros(shape=(X.shape[0],k))
        labels = np.zeros(shape=(X.shape[0]))
        for i in range(self.max_iter):
            prev_centroids = centroids
            for j in range(len(X)):
                distance[j,:] = norm(X[j,:]-prev_centroids, axis=1)
                cluster_index = np.argmin(distance[j,:], axis=0)
                labels[j] = cluster_index
            centroids = self.__calculate_centroid(X,labels,k)
            if norm(prev_centroids-centroids) <= self.tolerance:
                break
        return centroids, labels
        
        
        
        
        
        
        
        

        
        
        
        