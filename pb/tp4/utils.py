from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score


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

def generate_new_features(x, n):
    x_mod = x.copy()
    km = KMeans(n_clusters=n, random_state=0)
    km.fit(x_mod)
    centroids = km.cluster_centers_

    for index_c, centroid in centroids:
        number_columns = len(x_mod.columns)
        x_mod.insert(number_columns, f'F{index_c}', [], True)

    return x_mod







    