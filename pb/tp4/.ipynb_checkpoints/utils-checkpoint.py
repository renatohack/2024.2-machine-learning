#!pip install yellowbrick

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score


def visualize_silhoutte_plot(x_train, n):
    fig, ax = plt.subplots(n//2, 2, figsize=(15,8), squeeze=False)

    for i in range(2, n+1):
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=0)
        q, mod = divmod(i, 2)
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
        visualizer.fit(x_train)

def visualize_silhouette_avg(x_train, n):
    for i in range(2, n + 1):
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=0)
        cluster_labels = km.fit_predict(x_train)
        print('N Clusters:', i, 'Avg:', silhouette_score(x_train, cluster_labels))

def visualize_elbow(x_train):
    km = KMeans(random_state=0)
    visualizer = KElbowVisualizer(km, k=(2,10))
     
    visualizer.fit(x_train)
    visualizer.show()