#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(11)


# In[2]:


means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0)
K = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T


# In[3]:


def kmeans_display(X, label):
    K = np.amax(label) + 1
    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]
    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8)

    plt.axis('equal')
    plt.plot()
    plt.show()
    
kmeans_display(X, original_label)


# In[12]:


def random_point(X, K):
    return X[np.random.choice(X.shape[0], K, replace=False)]


# In[13]:


def get_distance(X, centers):
    D = cdist(X, centers) 
    return np.argmin(D, axis = 1)


# In[14]:


def update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xk = X[labels == k, :]
        centers[k,:] = np.mean(Xk, axis = 0)
    return centers


# In[20]:


def check_converged(centers, new_centers):
    return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers]))


# In[34]:


def visualize(X, centers, labels, K, title):
    plt.xlabel('x') 
    plt.ylabel('y') 
    plt.title(title)
    plt_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
 
    for i in range(K):
        data = X[labels == i] 
        plt.plot(data[:, 0], data[:, 1], plt_colors[i] + 'o', markersize = 4) 
        plt.plot(centers[i][0], centers[i][1],  plt_colors[i+4] + '^', markersize = 10) 

    plt.show()


# In[35]:


def main(init_centers, init_labels, X, K):
    centers = init_centers
    labels = init_labels
    times = 0
    while True:
        labels = get_distance(X, centers)
        visualize(X, centers, labels, K, 'Clustering at time = ' + str(times + 1))
        new_centers = update_centers(X, labels, K)
        if check_converged(centers, new_centers):
            break
        centers = new_centers
        visualize(X, centers, labels, K, 'Update center at time = ' + str(times + 1))
        times += 1
    return (centers, labels, times)


# In[36]:


init_centers = random_point(X, K)
print(init_centers)
init_labels = np.zeros(X.shape[0])
visualize(X, init_centers, init_labels, K, 'Get random cluster')
centers, labels, times = main(init_centers, init_labels, X, K)


# In[37]:


print(centers)


# #### Nhận xét: Tâm 3 cụm xấp xỉ với tâm khởi tạo ban đầu

# In[ ]:




