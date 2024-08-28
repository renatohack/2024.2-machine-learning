from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score
import pandas as pd
import math


def visualize_silhoutte_plot(x, n):
    fig, ax = plt.subplots(n//2, 2, figsize=(15,8), squeeze=False)

    for i in range(2, n+1):
        km = KMeans(n_clusters=i, random_state=0)
        q, mod = divmod(i, 2)
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
        visualizer.fit(x)

def visualize_silhouette_avg(x, n):
    for i in range(2, n + 1):
        km = KMeans(n_clusters=i, random_state=0)
        cluster_labels = km.fit_predict(x)
        print('N Clusters:', i, 'Avg:', silhouette_score(x, cluster_labels))

def visualize_silhouette(x, n):
    visualize_silhoutte_plot(x, n)
    visualize_silhouette_avg(x, n)

def visualize_elbow(x):
    km = KMeans(random_state=0)
    visualizer = KElbowVisualizer(km, k=(2,10))
     
    visualizer.fit(x)
    visualizer.show()

def calcular_distancia_euclidiana(p1, p2):
    distance = 0
    for index in range(len(p2)):
        distance += (p1[index] - p2[index]) ** 2
    return math.sqrt(distance)

def generate_new_features(x, n, centroids):
    number_rows = len(x)
    x_mod = pd.DataFrame(x.copy())
    
    for index_c in range(len(centroids)):
        centroid = centroids[index_c]
        number_columns = len(x_mod.columns)
        column_name = f'F{index_c}'

        # inserrir nova coluna
        x_mod.insert(number_columns, column_name, pd.Series([0]*number_rows, dtype=float), True)

        # paa cada linha, calcular distancia e preencher na coluna gerada
        for index, row in x_mod.iterrows():
            distance = calcular_distancia_euclidiana(row, centroid)
            #x_mod[column_name][index] = distance
            x_mod.loc[index, column_name] = distance

    return x_mod


def criar_features_dataset(x_train, x_test, n):
    km = KMeans(n_clusters=n, random_state=0)
    km.fit(x_train)
    centroids = km.cluster_centers_

    x_train_mod = generate_new_features(x_train, n, centroids)
    x_test_mod = generate_new_features(x_test, n, centroids)

    return x_train_mod, x_test_mod