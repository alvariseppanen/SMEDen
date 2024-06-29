# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import torch
import torch.nn as nn
import torch.nn.functional as F

class lior(nn.Module):

    def __init__(self):
        super(lior, self).__init__()

    def LIOR(self, inputs):
        B, H, W = inputs.shape[0], inputs.shape[-2], inputs.shape[-1]

        alpha = 0.16
        beta = 3.0
        k_min = 3
        sr_min = 0.04
        search = 5
        search_dim = search ** 2
        pad = int((search - 1) / 2)
        i_th_c = 0.0469 * 126445

        proj_range = inputs[:,[0],:,:]
        proj_xyz = inputs[:,1:4,:,:]
        proj_intensity = inputs[:,[4],:,:]
                 
        unfold_inputs = F.unfold(proj_xyz,
                            kernel_size=(search, search),
                            padding=(pad, pad))

        x_points = unfold_inputs[:, 0*search_dim:1*search_dim, :]
        y_points = unfold_inputs[:, 1*search_dim:2*search_dim, :]
        z_points = unfold_inputs[:, 2*search_dim:3*search_dim, :]
        xyz_points = torch.cat((x_points, y_points, z_points), dim=0)
        
        proj_xyz = torch.swapaxes(proj_xyz[None,:,:,:].flatten(start_dim=2), 0, 1)
        differences = xyz_points - proj_xyz
        differences = torch.linalg.norm(differences, dim=0)
        differences = differences[None,:,:]

        proj_range = proj_range.flatten(start_dim=2)
        sr_p_map = torch.zeros((proj_range.shape))
        sr_p_map[proj_range < sr_min] = sr_min
        sr_p_map[proj_range >= sr_min] = beta * (proj_range[proj_range >= sr_min] * alpha)

        radius_inliers = torch.count_nonzero(differences < sr_p_map, dim=1)
        predictions = torch.zeros((radius_inliers.shape))

        i_th = i_th_c / proj_range**2
        predictions[proj_intensity < i_th] = 1
        predictions[predictions * radius_inliers > k_min] = 0
        
        return predictions

    def forward(self, x):
        predictions = self.LIOR(x)

        return predictions
