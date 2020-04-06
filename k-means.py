from sklearn.datasets import load_iris, load_wine
import random
import numpy as np
from sklearn.metrics import adjusted_rand_score as ars

def k_means(X,k=2):
	y = [-1 for i in range(len(X))]
	y_last = y
	for i in range(k):
		rand = random.randint(0,len(X)-1-k)
		while 1 > 0:
			if y[rand] == -1:
				y[rand] = i
				break
			else:
				rand = (rand+1)%len(X)
	
	while (y_last != y) or (-1 in y):
		y_last = y
		cntr = centroids(y,X)
		for i in range(len(X)):
			y_pred = nearest_centroids_classifier(y, X[i], cntr)
			y[i] = y_pred[0] if y_pred[0] != -1 else y_pred[1]
	return y
	
def centroids(objects,coordinates):
	centers = {}
	for i in range(len(objects)):
		centers[objects[i]] = centers.get(objects[i],[])
		centers[objects[i]].append(coordinates[i])
	for i in centers:
		centers[i] = sum(centers[i])/len(centers)
	return centers

def nearest_centroids_classifier(y, x, center):
    centers = center
    distances = {i : calc_distance(x, centers[i]) for i in centers.keys()}
    distances = sorted(distances, key=distances.get)
    return distances[0:1]
    

def calc_distance(u, v):
    return np.sqrt(np.sum((u - v)**2))
  

data = load_iris()
X = data.data
y = data.target
y_pred = k_means(X,k=3)
print(y_pred)
print(ars(y,y_pred))



