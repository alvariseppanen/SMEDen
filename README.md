## Multi-Echo Denoising in Adverse Weather

![](https://github.com/alvariseppanen/4DenoiseNet/blob/main/self_demo.gif)


### Citation:

coming soon


### SnowyKITTI-dataset:

[Download](https://www.dropbox.com/s/o3r654cdzfl405d/snowyKITTI.zip?dl=0)


### Train:
```
cd networks
./self_train.sh -d root/snowyKITTI/dataset/ -a smednet.yml -l /your/log/folder/ -c 0
```

### Infer (pretrained model -m root/logs/2023-2-21-15:49/):
```
cd networks/train/tasks/semantic
python3 self_infer.py -d root/toy_snowyKITTI/dataset/ -m root/logs/2023-2-21-15:49/ -l /your/predictions/folder/ -s test
(-s = split)
```

### Evaluate:
```
cd networks/train/tasks/semantic
python3 snow_evaluate_iou.py -d root/toy_snowyKITTI/dataset/ -dc root/networks/train/tasks/semantic/config/labels/snowy-kitti.yaml -p /your/predictions/folder/ -s test
(-s = split)
```

### Visualize:
```
cd utils
python3 snow_visualize.py -d root/toy_snowyKITTI/dataset/ -c root/networks/train/tasks/semantic/config/labels/snowy-kitti.yaml -p /your/predictions/folder/ -s 22
(-s = sequence)
```

Thanks to [SalsaNext](https://github.com/TiagoCortinhal/SalsaNext) for providing some of the code! 
# SMEDNet
# SMEDNet
