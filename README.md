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

* ### Standard deviation of the MAE and RMSE
<img width="453" alt="image" src="https://github.com/user-attachments/assets/dd1110ab-3996-4a47-a727-7f33dbac2fd1"
     style="display: block; margin: auto;" />

We have conducted experiments on ETTm1, ETTm2, ETTh1, ETTh2, Weather, and Exchange with three different random seeds (i.e., 2024, 6666, 8888) and report the Mean and Standard deviation (Std) of the MSE and MAE in Table A.
We observe that the standard deviation of MSE and MAE is minimal ranging from 0.0014 to 0.0042. These results indicate that TimeKD exhibits strong robustness and stability across different random seeds.
