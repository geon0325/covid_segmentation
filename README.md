# Simple Epidemic Models with Segmentation Can Be Betterthan Complex Ones
Source code for the paper [Simple Epidemic Models with Segmentation Can Be Betterthan Complex Ones](https://github.com/geonlee0325/covid_segmentation), Geon Lee, Se-eun Yoon, Kijung Shin.

## Datasets
* The original datasets are available [here](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset).
* The processed datasets are available in [data folder](https://github.com/geonlee0325/covid_segmentation/tree/main/data).

## Requirements
To properly run the code, please install following required packages via:
```setup
pip install pykalman
pip install lmfit
```

## Running Demo
* For LLD and NLLD models, you can set the latent value (k) at line 21 in [nlds.py](https://github.com/geonlee0325/covid_segmentation/blob/main/code/nlds.py).
* You can run by excuting:
```setup
./run.sh
```
* There are 10 different *runtype*s which can be set in [run.sh](https://github.com/geonlee0325/covid_segmentation/blob/main/code/run.sh):
```setup
1. Fitting using LLD/NLLD (our segmentation scheme)
2. Forecasting using LLD/NLLD (our segmentation scheme)
3. Fitting using LLD/NLLD (incremental segmentation)
4. Fitting using LLD/NLLD (single segmentation)
5. Forecasting using LLD/NLLD (single segmentation)
6. Fitting with SIR (our segmentation scheme)
7. Forecasting with SIR (our segmentation scheme)
8. Fitting with SIR (single segmentation)
9. Forecasting with SIR (single segmentation)
10. Fitting with SIR (incrementatl segmentation)
```
## Contact Information
If you have any questions, please contact [Geon Lee](https://geonlee0325.github.io/).
