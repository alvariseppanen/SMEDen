# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import torch
import torch.nn as nn
import torch.nn.functional as F

class medror(nn.Module):

    def __init__(self, n_echoes=2):
        super(medror, self).__init__()
        kernel_knn_size = 3
        self.search = 9
        self.knn_nr = kernel_knn_size ** 2
        self.n_echoes = n_echoes

    def MEDROR(self, inputs):
        B, H, W = inputs.shape[0], inputs.shape[-2], inputs.shape[-1]
        search_dim = self.search ** 2
        pad = int((self.search - 1) / 2)

        first_points = (inputs[:, self.n_echoes:self.n_echoes+3, ...].clone())
        first_unfold_points = F.unfold(first_points,
                            kernel_size=(self.search, self.search),
                            padding=(pad, pad))

        first_unfold_points = first_unfold_points.view(B, 3, search_dim, H*W)

        predictions = torch.zeros((B, 0, H, W)).cuda()
        for echo in range(self.n_echoes):
            n_range = (inputs[:, echo:echo+1, ...].clone())
            n_points = (inputs[:, self.n_echoes+echo*3:self.n_echoes+echo*3+3, ...].clone())

            n_points = n_points.flatten(start_dim=2).unsqueeze(dim=2)
            n_distance = torch.linalg.norm(n_points - first_unfold_points, dim=1)
            n_knn_values, n_knn_index = n_distance.topk(self.knn_nr, dim=1, largest=False)

            n_knn_values[n_knn_values > n_range.flatten(start_dim=2) * 3 * 0.008] = 0 # 0.08 0.03, 0.02 0.12, 0.01 0.24, 0.005 0.30
            n_knn_values = torch.count_nonzero(n_knn_values, dim=1).unsqueeze(dim=1)
            n_prediction = (n_knn_values < 3).int() # 1 non-valid, 0 valid # 3
            # a hack for binary label
            n_prediction *= 2000
            n_prediction -= 1000

            n_prediction = n_prediction.view(B, 1, H, W)
            predictions = torch.cat((predictions, n_prediction), dim=1)

        return predictions

    def forward(self, x):
        predictions = self.MEDROR(x)

        return predictions
