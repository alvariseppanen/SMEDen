#!/usr/bin/env python3
import numpy as np
import glob, os
import argparse
import open3d as o3d

def key_func(x):
        return os.path.split(x)[-1]

seq = 4

strongest_root = '/home/seppanen/custom_data/newSTF/dataset/sequences/' + str(seq).zfill(2) + '/snow_velodyne/*.bin'
strongest_data = sorted(glob.glob(strongest_root), key=key_func)

last_root = '/home/seppanen/custom_data/newSTF/dataset/sequences/' + str(seq).zfill(2) + '/last_velodyne/*.bin'
last_data = sorted(glob.glob(last_root), key=key_func)

save_path = '/home/seppanen/custom_data/newSTF/dataset/sequences/'

for i in range(len(strongest_data)):

    strongest_points = np.fromfile(strongest_data[i], dtype=np.float32)
    strongest_points = strongest_points.reshape((-1, 4))
    strongest_points = strongest_points[:, :4]

    last_points = np.fromfile(last_data[i], dtype=np.float32)
    last_points = last_points.reshape((-1, 4))
    last_points = last_points[:, :4]

    combined = np.concatenate((strongest_points, last_points), axis=0)

    combined.astype('float32').tofile(save_path + str(seq).zfill(2) + '/combined_velodyne/' + str(i).zfill(6) + '.bin')

    pcd = o3d.t.geometry.PointCloud()

    pcd.point["positions"] = o3d.core.Tensor(combined[:, :3])
    pcd.point["intensities"] = o3d.core.Tensor(combined[:, 3][:, None])

    o3d.t.io.write_point_cloud(save_path + str(seq).zfill(2) + '/combined_velodyne_pcd/' + str(i).zfill(6) + '.pcd', pcd)

    #if i > 10: break
