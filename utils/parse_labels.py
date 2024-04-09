#!/usr/bin/env python3
import numpy as np
import glob, os
import argparse
import open3d as o3d
import json

def key_func(x):
        return os.path.split(x)[-1]

seq = 4

strongest_root = '/home/seppanen/custom_data/newSTF/dataset/sequences/' + str(seq).zfill(2) + '/snow_velodyne/*.bin'
strongest_data = sorted(glob.glob(strongest_root), key=key_func)

last_root = '/home/seppanen/custom_data/newSTF/dataset/sequences/' + str(seq).zfill(2) + '/last_velodyne/*.bin'
last_data = sorted(glob.glob(last_root), key=key_func)

save_path = '/home/seppanen/custom_data/newSTF/dataset/'

label_json = open('seq4-all.json')
label_data = json.load(label_json)

for i in range(len(label_data['dataset']['samples'])):
    
    label_name = label_data['dataset']['samples'][i]['name']
    labels = label_data['dataset']['samples'][i]['labels']['ground-truth']['attributes']['point_annotations']
    print(label_name)

    strongest_points = np.fromfile(strongest_data[i], dtype=np.float32)
    strongest_points = strongest_points.reshape((-1, 4))
    strongest_points = strongest_points[:, :4]

    last_points = np.fromfile(last_data[i], dtype=np.float32)
    last_points = last_points.reshape((-1, 4))
    last_points = last_points[:, :4]

    combined = np.concatenate((strongest_points, last_points), axis=0)

    #print(combined.shape)

    #combined.astype('float32').tofile(save_path + str(seq).zfill(2) + '/combined_velodyne/' + str(i).zfill(6) + '.bin')

    #pcd = o3d.t.geometry.PointCloud()
    #pcd.point["positions"] = o3d.core.Tensor(combined[:, :3])
    #pcd.point["intensities"] = o3d.core.Tensor(combined[:, 3][:, None])

    #o3d.t.io.write_point_cloud(save_path + str(seq).zfill(2) + '/combined_velodyne_pcd/' + str(i).zfill(6) + '.pcd', pcd)

    #if i > 10: break

    labels_np = np.array(labels).astype(np.int32)

    # 1 is snowfall, 0 is valid
    ''' convert to labels: 
        0: "valid"
        1: "noise"
        2: "substitute"
        3: "discarded"
        9: "unlabelled"
    '''

    '''half_shape = labels_np.shape[0] // 2
    strongest_mask = np.zeros(labels_np.shape).astype(np.int32)
    strongest_mask[:half_shape] = 1
    last_mask = np.logical_not(strongest_mask)
    substitute_mask = last_mask * np.logical_not(labels_np)
    labels_np[substitute_mask] = 2'''
    
    # can't define gt-substitutes here because multi_laserscan decides last echoes geometrically.
    # TODO: modify IoU eval so that predictions define the gt-substitutes 
    
    labels_np = labels_np.reshape((-1)).astype(np.int32)
    path = os.path.join(save_path, "sequences",
                        str(seq).zfill(2), "snow_labels", label_name[:6] + '.label')
    labels_np.tofile(path)