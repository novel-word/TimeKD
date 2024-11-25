## Dependencies

* Python 3.10
* PyTorch 2.1.2
* cuda 12.1
* torchvision 0.16.2

```bash
> conda env create -f env_ts.yaml
```

## Datasets
Datasets can be obtained from [TimesNet](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2)

## Usages
* ### Storage 

```bash
chmod +x Store.sh
./Store.sh
```

* ### Forecasting
   
```bash
chmod +x Fcst.sh
./Fcst.sh
```
