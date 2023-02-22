# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import torch
import torch.nn.functional as F

class KNN_search():

    def __init__(self, kernel_size=2, search=19):
        super(KNN_search, self).__init__()
        self.search = search
        self.knn_nr = kernel_size ** 2
        
    def KNNs(self, values, features):
        B, H, W = values.shape[0], values.shape[-2], values.shape[-1]
        search_dim = self.search ** 2
        pad = int((self.search - 1) / 2)
        
        unfold_values = F.unfold(values,
                            kernel_size=(self.search, self.search),
                            padding=(pad, pad))
        unfold_features = F.unfold(features,
                            kernel_size=(self.search, self.search),
                            padding=(pad, pad))

        n_values = torch.cat((unfold_values[:, search_dim*0:search_dim*1, ...].unsqueeze(dim=1),  
                              unfold_values[:, search_dim*1:search_dim*2, ...].unsqueeze(dim=1)), dim=1)
        
        c_values = values.flatten(start_dim=2).unsqueeze(dim=2)

        distance = torch.linalg.norm(c_values - n_values, dim=1)
        closest_value, closest_index = distance.topk(self.knn_nr, dim=1, largest=False)

        unfold_features = torch.gather(input=unfold_features, dim=1, index=closest_index)

        feature_knn = unfold_features.view(B, self.knn_nr, H, W)
        knn_value = closest_value.view(B, self.knn_nr, H, W)
        return feature_knn, knn_value