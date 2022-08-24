# Simple Epidemic Models with Segmentation Can Be Better than Complex Ones
Source code for the [PLOS ONE (2022)](https://journals.plos.org/plosone/) paper [Simple Epidemic Models with Segmentation Can Be Better than Complex Ones](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0262244), Geon Lee, Se-eun Yoon, Kijung Shin.
This was also presented in [epiDAMIK workshop in KDD 2021](https://epidamik.github.io/2021/index.html).

## Datasets
* The original datasets are available [here](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset).
* The processed datasets are available in [data folder](https://github.com/geonlee0325/covid_segmentation/tree/main/data).

## Requirements
To properly run the code, please install the following required packages via:
```setup
pip install pykalman
pip install lmfit
```

## Running Demo
* There are 10 different **runtype**s which can be set in [run.sh](https://github.com/geonlee0325/covid_segmentation/blob/main/code/run.sh):
```setup
1.  Fitting using LLD/NLLD (our segmentation scheme)
2.  Fitting using LLD/NLLD (single segmentation)
3.  Fitting using LLD/NLLD (incremental segmentation)
4.  Forecasting using LLD/NLLD (our segmentation scheme)
5.  Forecasting using LLD/NLLD (single segmentation)
6.  Fitting with SIR (our segmentation scheme)
7.  Fitting with SIR (single segmentation)
8.  Fitting with SIR (incremental segmentation)
9.  Forecasting with SIR (our segmentation scheme)
10. Forecasting with SIR (single segmentation)
```
* For **runtype**s 1, 2, 3, 4, and 5, execute:
```setup
./run.sh [runtype] [country] [output directory] [LLD or NLLD] [latent dimension k] [error rate (only for runtype 3)]
e.g., ./run.sh 1 japan ./ NLLD 3
```
* For **runtype**s 6, 7, 8, 9, and 10, execute:
```setup
./run.sh [runtype] [country] [output directory] [SIR] [error rate (only for runtype 8)]
e.g., ./run.sh 6 japan ./ SIR
```

## Contact Information
If you have any questions, please contact [Geon Lee](https://geonlee0325.github.io/).
