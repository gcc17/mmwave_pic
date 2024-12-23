import numpy as np
from .strategy import Strategy
from tqdm import tqdm

class KCenterGreedy(Strategy):
    def __init__(self, coreset_dataset, model):
        super(KCenterGreedy, self).__init__(coreset_dataset, model)
    
    def query(self, n):
        selected_indices = self.coreset_dataset.selected_indices
        unselected_indices = np.array([i for i in range(len(self.coreset_dataset.ori_dataset)) if i not in selected_indices])
        
        # extract all feature embeddings
        feature_embeddings = self.model.get_feature_embeddings(self.coreset_dataset.ori_dataset)
        feature_embeddings = feature_embeddings.detach().cpu().numpy()
        
        dist_mat = np.matmul(feature_embeddings, feature_embeddings.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(self.coreset_dataset.ori_dataset), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)
        
        mat = dist_mat[unselected_indices, :][:, selected_indices]
        new_indices = []
        for i in tqdm(range(n), ncols=100):
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = unselected_indices[q_idx_]
            selected_indices = np.append(selected_indices, q_idx)
            new_indices.append(q_idx)
            mat = np.delete(mat, q_idx_, axis=0)
            unselected_indices = np.delete(unselected_indices, q_idx_, axis=0)
            mat = np.append(mat, dist_mat[unselected_indices, q_idx][:, None], axis=1)
        
        new_indices = np.array(new_indices)
        return new_indices
    
    