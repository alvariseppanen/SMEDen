#!/usr/bin/env python3
import numpy as np
import glob, os
import argparse

def key_func(x):
        return os.path.split(x)[-1]

def convert(dataset, new):

    snow_indices = [[0,100], [480, 570], [585, 600], [620, 628], [1063, 1083], [1100, 1232], [1288, 1320], [1815, 1990], [2224, 2230], [2356, 2368], [2594, 2615], [2727, 2730],
                    [4036, 4050], [4058, 4158], [4161, 4322], [4357, 4866], [4874, 4886], [5504, 5618], [6000, 6125], [6408, 6502], [6532, 6550], [6763, 6844], [6853, 6866],
                    [6876, 6880], [7201, 7293], [7360, 7364], [7764, 7949], [7955, 7961], [8498, 8541], [8605, 8607], [8928, 8936], [8981, 8987], [9061, 9067], [9137, 9157],
                    [9215, 9297], [10123, 10132], [10211, 10219], [10235, 10245], [10736, 10750], [10780, 10796], [10910, 10913], [10919, 10945], [10954, 10971], 
                    [11000, 11262], [11289, 11333], [11407, 11667]]

    strongest_root = dataset + 'SeeingThroughFogCompressed/lidar_hdl64_strongest/lidar_hdl64_strongest/*.bin'
    last_root = dataset + 'SeeingThroughFogCompressed/lidar_hdl64_last/lidar_hdl64_last/*.bin'
    new_root = os.path.join(new, "new_STF", "dataset", "sequences")

    os.makedirs(new_root)
    seqs = ['00', '01', '02', '03', '04', '05']
    for item in seqs:
        path = os.path.join(new_root, item)
        os.mkdir(path)
        path = os.path.join(new_root, item, "snow_velodyne")
        os.mkdir(path)
        path = os.path.join(new_root, item, "last_velodyne")
        os.mkdir(path)

    strongest_data = sorted(glob.glob(strongest_root), key=key_func)
    last_data = sorted(glob.glob(last_root), key=key_func)
    
    new_i = 0
    new_sequence = 0
    j = 0
    iter = 0
    for j in range(len(snow_indices)):
        for i in range(snow_indices[j][0], snow_indices[j][1]):
            if iter % 500 == 0 and iter > 0:
                print("new seq")
                new_sequence += 1
                new_i = 0
            
            strongest_points = np.fromfile(strongest_data[i], dtype=np.float32)
            strongest_points = strongest_points.reshape((-1, 5))
            strongest_points = strongest_points[:, :4]
            
            last_points = np.fromfile(last_data[i], dtype=np.float32)
            last_points = last_points.reshape((-1, 5))
            last_points = last_points[:, :4]
            
            strongest_points.astype('float32').tofile(new_root + '/' + str(new_sequence).zfill(2) + '/snow_velodyne/' + str(new_i).zfill(6) + '.bin')
            last_points.astype('float32').tofile(new_root + '/' + str(new_sequence).zfill(2) + '/last_velodyne/' + str(new_i).zfill(6) + '.bin')
            
            iter += 1
            new_i += 1
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser("./create_new_stf.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='path to STF dataset. No Default',
    )
    parser.add_argument(
        '--new', '-n',
        type=str,
        default=os.path.expanduser("~") + '/new_STF/',
        help='Directory to put the new dataset.'
    )
    FLAGS, unparsed = parser.parse_known_args()

    convert(FLAGS.dataset, FLAGS.new)