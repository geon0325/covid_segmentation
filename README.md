# Simple Epidemic Models with Segmentation Can Be Betterthan Complex Ones
Source code for the paper [Simple Epidemic Models with Segmentation Can Be Betterthan Complex Ones](https://github.com/geonlee0325/covid_segmentation), Geon Lee, Se-eun Yoon, Kijung Shin.

## Datasets
* The original datasets are available [here](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset).
* The processed datasets are available in [data folder](https://github.com/geonlee0325/covid_segmentation/tree/main/data).
 
## Running Demo
* For LLD and NLLD models, you can set the latent value (k) at line 21 in [nlds.py](https://github.com/geonlee0325/covid_segmentation/blob/main/code/nlds.py).
* You can run by
```setup
./run.sh
```
* aaa
You can run demo with the sample dataset (dblp_graph.txt).
1. To run **MoCHy-E**, type 'run_exact.sh'.
2. To run *parallelized* **MoCHy-E**, type 'run_exact_par.sh'.
3. To run **MoCHy-A**, type 'run_approx_ver1.sh'.
4. To run **MoCHy-A+**, type 'run_approx_ver2.sh'.
5. To run *parallelized* **MoCHy-A+**, type 'run_approx_ver2_par.sh'.
6. To run *memory-bounded* **MoCHy-A+**, type 'run_approx_ver2_memory.sh'.

## Terms and Conditions
If you use this code as part of any published research, please acknowledge our VLDB 2020 paper.
```
@article{lee2020hypergraph,
  title={Hypergraph Motifs: Concepts, Algorithms, and Discoveries},
  author={Lee, Geon and Ko, Jihoon and Shin, Kijung},
  journal={Proceedings of the VLDB Endowment},
  year={2020},
  publisher={VLDB Endowment}
}
```

## Contact Information
If you have any questions, please contact [Geon Lee](geonlee0325@kaist.ac.kr).
