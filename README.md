## Self-supervised multi-echo point cloud denoising in snowfall, [Publication](https://www.sciencedirect.com/science/article/pii/S0167865524002101)

![](https://github.com/alvariseppanen/SMEDNet/blob/main/self_demo.gif)

### Citation:

@article{seppanen2024self,
  title={Self-supervised multi-echo point cloud denoising in snowfall},
  author={Sepp{\"a}nen, Alvari and Ojala, Risto and Tammi, Kari},
  journal={Pattern Recognition Letters},
  year={2024},
  publisher={Elsevier}
}

### Environment: 

- Python 3.8.10
- CUDA 11.6
- PyTorch 1.12.1+cu116
- Numpy 1.23.3

### Datasets:

- [Download SnowyKITTI](https://github.com/alvariseppanen/4DenoiseNet)
- [Download STF](https://github.com/princeton-computational-imaging/SeeingThroughFog)

Collect corrupted point clouds from STF:

```
cd utils
python3 create_new_stf.py -d root/STF_dataset/ -n root_for_new_dataset/
```

### Train:
```
cd networks
./self_train.sh -d root/snowyKITTI/dataset/ -a smednet.yml -l /your/log/folder/ -c 0
./multi_self_train.sh -d root/new_STF/dataset/ -a smednet.yml -l /your/log/folder/ -c 0 
```

### Infer (pretrained singe-echo model -m root/logs/2023-2-21-15:49/, multi-echo model -m root/logs/2023-2-27-13:11/):
```
cd networks/train/tasks/semantic
python3 self_infer.py -d root/snowyKITTI/dataset/ -m root/logs/2023-2-21-15:49/ -l /your/predictions/folder/ -s test
python3 multi_self_infer.py -d root/new_STF/dataset/ -m root/logs/2023-2-27-13:11/ -l /your/predictions/folder/ -s test
(-s = split)
```

### Evaluate:
```
cd networks/train/tasks/semantic
python3 evaluate_iou.py -d root/snowyKITTI/dataset/ -dc root/networks/train/tasks/semantic/config/labels/snowy-kitti.yaml -p /your/predictions/folder/ -s test
(-s = split)
```

### Visualize:
```
cd utils
single-echo:
python3 visualize.py -d root/snowyKITTI/dataset/ -c root/networks/train/tasks/semantic/config/labels/snowy-kitti.yaml -p /your/predictions/folder/ -s 22
multi-echo:
python3 visualize.py -d root/new_STF/dataset/ -c root/networks/train/tasks/semantic/config/labels/stf.yaml -p /your/predictions/folder/ -s 4 -me 
(-me = multi-echo)
```
