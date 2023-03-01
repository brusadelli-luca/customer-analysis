from functions import dist_calc
from random import randint

import pandas as pd
import matplotlib.pyplot as plt


from statistics import mean

class kmeans():

    def __init__(self):
        
        self.c = []
        self.unique_y = []
        self.clusters = []
    

    # 1. Decide how many clusters you want, i.e. choose k

    def fit(self, X, k=3, plot=False):
          
        # 2. Randomly assign a centroid to each of the k clusters
        
        #Sélection aléatoire de 3 points dans la fenêtre des X = centroïdes de départ

        x0_min, x0_max, x1_min, x1_max = round(min(X[:, 0])), round(max(X[:, 0])), round(min(X[:, 1])), round(max(X[:, 1]))
        
        centroids = [[randint(x0_min, x0_max), randint(x1_min, x1_max)] for c in range(k)]
        centroids_list = [centroids]

        
        # Repeat steps 3-5 until the centroids do not change position

        #Choix du nombre d'itérations
        steps = 10

        for step in range(steps):

            # 3. Calculate the distance of all observation to each of the k centroids

            dists = [[dist_calc(P, c) for c in centroids_list[-1]] for P in X]


            # 4. Assign observations to the closest centroid

            y = [d.index(min(d)) for d in dists]      


            # 5. Find the new location of the centroid by taking the mean of all the observations in each cluster

            df_X = pd.DataFrame(X, columns = ['x0', 'x1'])
            df_y = pd.DataFrame(y, columns = ['y_pred'])

            df = pd.concat((df_X, df_y), axis = 1)

            unique_y = list(dict.fromkeys(y))
            unique_y.sort()

            new_centroids = [[round(mean(df[df['y_pred'] == o].x0), 2), round(mean(df[df['y_pred'] == o].x1), 2)] for o in unique_y]

            centroids_list.append(new_centroids)

        self.c = new_centroids
        self.unique_y = unique_y
        self.clusters = [df[df['y_pred'] == i].reindex(columns = ['x0','x1']).to_numpy() for i in unique_y]

        # Plot
        if plot:
            dfR = pd.DataFrame(centroids_list)

            c_list = [dfR[u].to_list() for u in unique_y]

            for i in range(len(unique_y)):
                plt.plot([c[0] for c in c_list[i]], [c[1] for c in c_list[i]])

            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")


    def elbow(self, X, K):

        result = []

        for k in range(1, K + 1):

            self.fit(X, k)
            
            inertie = 0
            for i in range(len(self.unique_y)):
                inertie = inertie + round(sum([dist_calc(P, self.c[i]) for P in self.clusters[i]]), 2)

            result.append(round(inertie, 2))
        
        return result


    def predict(self, X):
        
        dists = [[dist_calc(P, c) for c in self.c] for P in X]
    
        y = [d.index(min(d)) for d in dists]

        return y