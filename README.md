# ExpressGNN

This is an implementation of the ExpressGNN proposed in the paper "Efficient Probabilistic Logic Reasoning with Graph Neural Networks".

## Requirements
- python 3.7
- pytorch 1.1
- scikit-learn
- networkx
- tqdm

## Quick Start
The following command starts the inference on the Kinship-S1 dataset on GPU:
```
python -m main.train -data_root data/kinship/S1 -slice_dim 8 -batchsize 16 -use_gcn 1 -embedding_size 64 -gcn_free_size 32 -load_method 0 -exp_folder exp -exp_name kinship -device cuda
```

To run ExpressGNN on the FB15K-237 dataset on GPU, use the follwoing command line:
Free Parameters + GNN, Free parameters dim = free_dim, GNN parameters dim = embedding_size - free_dim
```
python -m main.train -slice_dim 16 -batchsize 16 -num_hops 2 -patience 20 -lr_decay_patience 100 -entropy_temp 1 -load_method 1 -exp_folder exp -device cuda -rule_filename cleaned_rules_weight_larger_than_0.9.txt -data_root data/fb15k-237 -embedding_size 128 -gcn_free_size 127  -exp_name fb15k-237 -mylambda 1 -exp_mode 1
```

GNN parameters, Free parameters dim = 0, GNN parameters dim = embedding_size 
```
python -m main.train -slice_dim 16 -batchsize 16 -num_hops 2 -patience 20 -lr_decay_patience 100 -entropy_temp 1 -load_method 1 -exp_folder exp -device cuda -rule_filename cleaned_rules_weight_larger_than_0.9.txt -data_root data/fb15k-237 -embedding_size 128 -exp_name fb15k-237 -mylambda 1 -exp_mode 2
```

GNN parameters, Free parameters dim = embedding_size, GNN parameters dim = 0 
```
python -m main.train -slice_dim 16 -batchsize 16 -num_hops 2 -patience 20 -lr_decay_patience 100 -entropy_temp 1 -load_method 1 -exp_folder exp -device cuda -rule_filename cleaned_rules_weight_larger_than_0.9.txt -data_root data/fb15k-237 -embedding_size 128 -exp_name fb15k-237 -mylambda 1 -exp_mode 3
```
