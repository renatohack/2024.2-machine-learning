from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score
import pandas as pd
import math
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
import seaborn as sns


def visualize_silhouette(x, n):
    fig, ax = plt.subplots(n//2, 2, figsize=(30,16), squeeze=False)
    silhouette_scores = []
    range_k = range(2, n+1)
    
    for i in range_k:
        km = KMeans(n_clusters=i, random_state=0)
        clusters = km.fit_predict(x)
        score = silhouette_score(x, clusters)
        silhouette_scores.append(score)
        
        print('N Clusters:', i, 'Avg:', silhouette_score(x, clusters))

        q, mod = divmod(i, 2)
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
        visualizer.fit(x)

    return silhouette_scores, range_k



def visualize_elbow(x, n):
    km = KMeans(random_state=0)
    visualizer = KElbowVisualizer(km, k=(2,n))
    visualizer.fit(x)
    visualizer.show()


def visualize_sil_score_per_cluster(silhouette_scores, range_k):
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, marker='o')
    plt.title("Silhouette Score para diferentes valores de K")
    plt.xlabel("Número de Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.show()


def calcular_distancia_euclidiana(p1, p2):
    distance = 0
    for index in range(len(p2)):
        distance += (p1[index] - p2[index]) ** 2
    return math.sqrt(distance)

def generate_new_features(x, centroids):
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

    x_mod.columns = x_mod.columns.astype(str)

    return x_mod


def criar_features_dataset(x_train, x_test, centroids):
    x_train_mod = generate_new_features(x_train, centroids)
    x_test_mod = generate_new_features(x_test, centroids)

    return x_train_mod, x_test_mod


def show_estimator_results(grid, x_test, y_test):
    print(f"Melhores parâmetros GridSearch: {grid.best_params_}")
    print(f'Score GridSearch: {grid.score(x_test, y_test)}')
    
    print()
    print('Classification Report')
    y_pred = grid.predict(x_test)
    print(classification_report(y_test, y_pred))

    return y_pred


def show_cm(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()