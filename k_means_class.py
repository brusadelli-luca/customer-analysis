from functions import dist_calc
from random import randint

import pandas as pd

from statistics import mean

class kmeans():

    def __init__(self):
        
        self.c = []
    

    # 1. Decide how many clusters you want, i.e. choose k

    def fit(self, X, k=3):
          
        # 2. Randomly assign a centroid to each of the k clusters
            
        x_min, x_max, y_min, y_max = round(min(X[:, 0])), round(max(X[:, 0])), round(min(X[:, 1])), round(max(X[:, 1]))
        centroids = [[randint(x_min, x_max), randint(y_min, y_max)] for c in range(k)]
        result = [centroids]

        
        # Repeat steps 3-5 until the centroids do not change position

        steps = 10

        centroids = [[randint(x0_min, x0_max), randint(x1_min, x1_max)] for c in range(k)]

        centroids_list = [centroids]

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


        # Plot

        dfR = pd.DataFrame(centroids_list)

        c_list = [dfR[u].to_list() for u in unique_y]

        for i in range(len(unique_y)):
            plt.plot([c[0] for c in c_list[i]], [c[1] for c in c_list[i]])

        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")


    def predict(self):
        0
