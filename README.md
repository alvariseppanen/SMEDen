## Self-supervised multi-echo point cloud denoising in snowfall, [Publication](https://pdf.sciencedirectassets.com/271524/1-s2.0-S0167865524X00082/1-s2.0-S0167865524002101/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEDYaCXVzLWVhc3QtMSJGMEQCIBYCrPVWgpibboWX7wsxQI%2FeuixkS6KKrXJfu514mla3AiBrPvOMRWtBPMHPqurA7iPtYl83mlv2ezI1zK%2FFOPCT0Cq8BQj%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAUaDDA1OTAwMzU0Njg2NSIMUGwxzBy6Z4%2FTK5%2ByKpAFyf9zziIog5h31NdRQdodO%2Fcyvcm9tvfrXvM7R9k7aFC2CTgEDlJGgxJZrsr17otBe8txPh6IeoG%2B0S%2B7WLXkdBwQLSHZH1u0xUiL8qbp8RfUOVeyXW5f%2FTOcvuxkTfaFdJctIVr6zjt7HcymoDnGyFTKm2JpCE4ltAOskbQ7XgXkcSc89zhwbH87TgumGlkW5In4rRlIhMFeyjRX4vuwy2AGs0A21XyzhiswI40rmQrVYEjgYxvHijcrnPiXUD2leW4yxaVik8JWl6KGMzbHh5EE4OxXtMiVV4APzYRJB3z5Qh%2FpZhLLEKSneVSkEJArc6%2B3e25jFBizZ6r3uG%2FCiYa89ZrzntoD%2B1uhOYcOdiYikwUhnkFpT2H%2Fvr%2B4S7f4XzUMAqG4frNtQcMyKWzAYTXKEX%2B3j0UUz%2F1lfn2t%2FdkqDmOBE9pELFC1M%2FCRuj%2BYkh73nO1lMpMSs0yytXjqdzP8PUEbsTAnjb3Auf3w89yRrEHZP2rHfyH0GNaC3wQ2xUHzUgSdh5GCP24LRcoQihhmRL0YZRPD49ZFViB0idiL2pAN1lGpt%2BnK%2BbwT9aJo2aRDDtsbRuHgW3iWiPmUaq4ydy0U%2Fypii6ITVA3AoE6mPC2eiZeOeKRUYSsT3DwLhLYTpy%2B8vXjyISdq1NXbKm4xtWT5NWV8ev11YM4IQmUK1cf22a2rhZmk81u3FXPAJ4lx%2BdGOMsWY%2F8RS8bgRgBEPtwOHv0kW8mCwJg8gNgBMxiTqrJzdnnwm3yoiGOq73FJQs1wPB7QkOrqtrbSQhNsIdcUnKHgQ%2BekLzo8TrsqYha1wVhFtwagrI3F2PJEVBuIYSmtIik06zrLtGzprQveV3i2HKNbY0S5xVIdoXVww49fitAY6sgGlpAtWlODPnnbILIjqrkeM25wp%2Fp6AOAbdLOWGhLDKKKSSB3GtB9myji90Ylnyu6u8c10gIq6GXvl7B1Mfnlz2BbUzirc4zgNBV34bBrddJivjJSSn2L5Y5zf7QihPpwCQ6BhT7vqxsUJzHAboMyBkbCE2sVTINn9WvXunMWr27ApzYDonDBidOExzaa1VfVWTiBJSwLWHproV9ZQvMWc3VEoYNsoDwZt0tABzaGk0NeOW&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240718T055302Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYYS26URPN%2F20240718%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=396ed386cf8a3c9023512d4d7ff9225ecc1d9983f111d1316fca5bcc13e9621e&hash=fcede497906e6629f67ee2420710d3f931d6d7f2cd57bb53a964f0f0f1b991b3&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0167865524002101&tid=spdf-70382d91-7bc3-4df6-a1dc-35a6470b22bb&sid=c8220e193565d14447998277fde2be0bd2c3gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=03135c0652565c525d&rr=8a503584ee692ed2&cc=au)

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
