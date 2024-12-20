import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans


class KMeansSampling(Strategy):
    def __init__(self, coreset_dataset, model):
        super(KMeansSampling, self).__init__(coreset_dataset, model)
    
    def query(self, n):
        # for all original data points. compute embeddings
        self.model.eval()
        feature_embeddings = self.model.get_feature_embeddings(self.coreset_dataset.ori_dataset)
        feature_embeddings = feature_embeddings.detach().cpu().numpy()
        # perform k-means clustering
        cluster_learner = KMeans(n_clusters=n, random_state=0)
        cluster_learner.fit(feature_embeddings)
        cluster_idxs = cluster_learner.predict(feature_embeddings)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = np.linalg.norm(feature_embeddings - centers, axis=1)
        q_idxs = np.array([np.arange(feature_embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])
        
        self.coreset_dataset.set_indices(q_idxs)
        return q_idxs
        