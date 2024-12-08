# Shrinkage-based Feature Selection Methods in Statistical Learning for High Dimensional Large-scale Data

## Simulations 
The script to run each method for simulation is outlined below. 
- [LASSOnet](https://github.com/nianyaodu/ece695_proj/blob/main/scripts/run_lassonet_simulations.py), which is modified from [lassonet package](https://github.com/lasso-net/lassonet)
- [HSIC LASSO](https://github.com/nianyaodu/ece695_proj/blob/main/scripts/run_hsic_lasso_simulations.py), which is modified from [pyHSICLasso package](https://github.com/riken-aip/pyHSICLasso)
- [GRNN](https://github.com/nianyaodu/ece695_proj/blob/main/scripts/run_grnn_simulations.py), which is modified from [pyGRNN package](https://github.com/federhub/pyGRNN)
- [SA](https://github.com/nianyaodu/ece695_proj/blob/main/scripts/run_sequantial_attention.py), which is modified from [sequantial_attention package](https://github.com/google-research/google-research/tree/master/sequential_attention)
- [LLY](https://github.com/nianyaodu/ece695_proj/blob/main/scripts/run_sequantial_attention.py), which is modified from [sequantial_attention package](https://github.com/google-research/google-research/tree/master/sequential_attention)
- [funGCN](https://github.com/nianyaodu/ece695_proj/blob/main/src/funGCN/expes/fgcn/run_fgcn_simulations.py), which is modified from [funGCN package](https://github.com/IBM/funGCN)


## Experiments 
The script to run each method for experiments is outlined below. 
- [LASSOnet](https://github.com/nianyaodu/ece695_proj/blob/main/scripts/run_lassonet.py), which is modified from [lassonet package](https://github.com/lasso-net/lassonet)
- [HSIC LASSO](https://github.com/nianyaodu/ece695_proj/blob/main/scripts/run_hsic_lasso.py), which is modified from [pyHSICLasso package](https://github.com/riken-aip/pyHSICLasso)
- [GRNN](https://github.com/nianyaodu/ece695_proj/blob/main/scripts/run_grnn.py), which is modified from [pyGRNN package](https://github.com/federhub/pyGRNN)
- [SA](https://github.com/nianyaodu/ece695_proj/blob/main/scripts/run_sequantial_attention_simulations.py), which is modified from [sequantial_attention package](https://github.com/google-research/google-research/tree/master/sequential_attention)
- [LLY](https://github.com/nianyaodu/ece695_proj/blob/main/scripts/run_sequantial_attention_simulations.py), which is modified from [sequantial_attention package](https://github.com/google-research/google-research/tree/master/sequential_attention)


To run LASSOnet, HSIC LASSO, GRNN, and funGCN, create an environment according to the corresponding yaml file in envs folder, change the output directory in the script and run the following command:

```bash
python run run_lassonet.py
```
To run SA and LLY, first download the dataset according to [this link](https://github.com/google-research/google-research/tree/master/sequential_attention/sequential_attention/experiments/datasets), and then following the steps above. 

## License

[MIT](https://choosealicense.com/licenses/mit/)
