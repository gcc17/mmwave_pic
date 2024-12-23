import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans
from copy import deepcopy

class KMeansSampling(Strategy):
    def __init__(self, coreset_dataset, model):
        super(KMeansSampling, self).__init__(coreset_dataset, model)
    
    def query(self, n):
        selected_indices = self.coreset_dataset.selected_indices
        unselected_indices = np.array([i for i in range(len(self.coreset_dataset.ori_dataset)) if i not in selected_indices])
        # obtain unselected feature embeddings
        unselected_coreset = deepcopy(self.coreset_dataset)
        unselected_coreset.selected_indices = unselected_indices
        self.model.eval()
        feature_embeddings = self.model.get_feature_embeddings(unselected_coreset)
        feature_embeddings = feature_embeddings.detach().cpu().numpy()
        # perform k-means clustering
        cluster_learner = KMeans(n_clusters=n, random_state=0)
        cluster_learner.fit(feature_embeddings)
        cluster_idxs = cluster_learner.predict(feature_embeddings)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = np.linalg.norm(feature_embeddings - centers, axis=1)
        q_idxs_ = np.array([np.arange(feature_embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])
        # convert query indices to original indices
        q_idxs = unselected_indices[q_idxs_]
        
        return q_idxs

    