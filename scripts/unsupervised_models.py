import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


def principal_component_analysis(dragon_subset):
    '''

    :param dragon_subset: Input Feature for Dragon data
    :return: Variance Captured across Principal Components
    '''
    pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=200))])

    X_PC = pipeline.fit_transform(dragon_subset)
    df_pc = pd.DataFrame(X_PC, columns=['PC_{}'.format(_) for _ in range(1, 201)])[['PC_1', 'PC_2']]
    df_pc.to_csv('df_principal_components.csv', index=False)
    df_variance = pd.DataFrame({'principal_component': ['PC_{}'.format(_) for _ in range(1, 201)],
                                'variance_captured': [round(_ * 100, 3) for _ in
                                                      pipeline['pca'].explained_variance_ratio_]})
    df_variance.to_csv('data/df_percent_variance_captured_in_PCA.csv', index=False)


def kmeans(dragon_subset, target):
    '''

    :param dragon_subset: Input Feature for Dragon data
    :param target: Output Feature (Density)
    :return: Data-Frame with Clusters
    '''
    scaler_ = StandardScaler()
    scaler_.fit(dragon_subset)
    x_scaled = scaler_.transform(dragon_subset)

    k_means = KMeans()
    visualizer = KElbowVisualizer(k_means, k=(2, 11))

    visualizer.fit(x_scaled)
    visualizer.show()

    kmeans_ = KMeans(n_clusters=4, random_state=42).fit(x_scaled)
    df_cluster = pd.concat([target, pd.DataFrame({'label': kmeans_.labels_})], ignore_index=False, axis=1)
    df_cluster.to_csv('data/df_cluster_allocation.csv', index=False)
