
Self-Supervised Graph Transformer on Large-Scale Molecular Data
===
This is a Pytorch implementation of the paper: Self-Supervised Graph Transformer on Large-Scale Molecular Data. 

## Requirements
 * Python 3.6.8
 * For the other packages, please refer to the `requirements.txt`. To resolve  `PackageNotFoundError`, please add the following channels before creating the environment. 
 ```bash
    conda config --add channels pytorch
    conda config --add channels rdkit
    conda config --add channels conda-forge
    conda config --add channels rmg
 ```
You can just execute following command to create the conda environment.
```
conda create --name chem --file requirements.txt
```
 * We also provide the Dockerfile to build the environment, please refer to the `Dockerfile` for more details. 

## Pretained Model Download
We provide the pretrained models used in paper. 
   - [GROVER<sub>base</sub>](https://ai.tencent.com/ailab/ml/ml-data/grover-models/pretrain/grover_base.tar.gz)
   - [GROVER<sub>large</sub>](https://ai.tencent.com/ailab/ml/ml-data/grover-models/pretrain/grover_large.tar.gz) 


## Usage
The whole framework supports **pretraining**, **finetuning**, **prediction**, **fingerprint generation**, and **evaluation** functions. 

### Pretraining
Pretrain `GTransformer` model given the unlabelled molecular data.  
#### Data Preparation
We provide an input example of unlabelled molecular data at `exampledata/pretrain/tryout.csv`. 
##### Semantic Motif Label Extraction
The semantic motif label is extracted by `scripts/save_feature.py` with feature generator `fgtasklabel`.
```bash
python scripts/save_features.py --data_path exampledata/pretrain/tryout.csv  \
                                --save_path exampledata/pretrain/tryout.npz   \
                                --features_generator fgtasklabel \
                                --restart
```
**Contributing guide**: you are welcomed to register your own feature generator to add more semantic motif for the graph-level prediction task. For more details, please refer to `grover/data/task_labels.py`.

##### Atom/Bond Contextual Property (Vocabulary)
The atom/bond Contextual Property (Vocabulary) is extracted by `scripts/build_vocab.py`.
 ```bash
python scripts/build_vocab.py --data_path exampledata/pretrain/tryout.csv  \
                              --vocab_save_folder exampledata/pretrain  \
                              --dataset_name tryout
 ```
The outputs of this script are vocabulary dicts of atoms and bonds, `tryout_atom_vocab.pkl` and 
`tryout_bond_vocab.pkl`, respectively. For more options for contextual property extraction, please refer to `scripts/build_vocab.py`.

##### Data Splitting
To accelerate the data loading and reduce the memory cost in the multi-gpu pretraining scenario, the unlabelled molecular 
data need to be spilt into several parts using `scrpits/split_data.py`.

**Note**: This step is required for single-gpu pretraining scenario as well.
```bash
python scripts/split_data.py --data_path exampledata/pretrain/tryout.csv  \
                             --features_path exampledata/pretrain/tryout.npz  \
                             --sample_per_file 100  \
                             --output_path exampledata/pretrain/tryout
```
It's better to set a larger `sample_per_file` for the large dataset. 

The output dataset folder will look like this:
```
tryout
  |- feature # the semantic motif labels
  |- graph # the smiles
  |- summary.txt
```

#### Running Pretraining on Single GPU
**Note:** There are more hyper-parameters which can be tuned during pretraining. Please refer to `add_pretrain_args` in`util/parsing.py` .
```bash
python main.py pretrain \
               --data_path exampledata/pretrain/tryout \
               --save_dir model/tryout \
               --atom_vocab_path exampledata/pretrain/tryout_atom_vocab.pkl \
               --bond_vocab_path exampledata/pretrain/tryout_bond_vocab.pkl \
               --batch_size 32 \
               --dropout 0.1 \
               --depth 5 \
               --num_attn_head 1 \
               --hidden_size 100 \
               --epochs 3 \
               --init_lr 0.0002 \
               --max_lr 0.0004 \
               --final_lr 0.0001 \
               --weight_decay 0.0000001 \
               --activation PReLU \
               --backbone gtrans \
               --embedding_output_type both
```

#### Running Pretraining on Multiple GPU
We have implemented distributed pretraining on multiple GPU using `horovod`. To start the distributed pretraining, please refer to [this link](https://github.com/horovod/horovod/blob/master/docs/running.rst). 
To enable the multi-GPU training of the pretraining model, `--enable_multi_gpu` flag should be proposed in the above command line.


### Training & Finetuning
The finetune dataset is organized as a `.csv` file. This file should contain a column named as `smiles`. 
#### (Optional) Molecular Feature Extraction
Given a labelled molecular dataset, it is possible to extract the additional molecular features in order to train & finetune the model from the existing pretrained model. The feature matrix is stored as `.npz`. 
``` bash
python scripts/save_features.py --data_path exampledata/finetune/bbbp.csv \
                                --save_path exampledata/finetune/bbbp.npz \
                                --features_generator rdkit_2d_normalized \
                                --restart 
```


#### Finetuning with Existing Data
Given the labelled dataset and the molecular features, we can use `finetune` function to finetunning the pretrained model. 

**Note:** There are more hyper-parameters which can be tuned during finetuning. Please refer to `add_finetune_args` in`util/parsing.py` .

```
python main.py finetune --data_path exampledata/finetune/bbbp.csv \
                        --features_path exampledata/finetune/bbbp.npz \
                        --save_dir model/finetune/bbbp/ \
                        --checkpoint_path model/tryout/model.ep3 \
                        --dataset_type classification \
                        --split_type scaffold_balanced \
                        --ensemble_size 1 \
                        --num_folds 3 \
                        --no_features_scaling \
                        --ffn_hidden_size 200 \
                        --batch_size 32 \
                        --epochs 10 \
                        --init_lr 0.00015
```
The final finetuned model is stored in `model/bbbp` and will be used in the subsequent prediction and evaluation tasks.  

### Prediction
Given the finetuned model, we can use it to make the prediction of the target molecules. The final prediction is made by the averaging the prediction of all sub models (num_folds * ensemble_size).

#### (Optional) Molecular Feature Extraction
**Note**: If the finetuned model uses the molecular feature as input, we need to generate the molecular feature for the target molecules as well. 
``` bash
python scripts/save_features.py --data_path exampledata/finetune/bbbp.csv \
                                --save_path exampledata/finetune/bbbp.npz \
                                --features_generator rdkit_2d_normalized \
                                --restart 
```

#### Prediction with Finetuned Model
``` bash
python main.py predict --data_path exampledata/finetune/bbbp.csv \
               --features_path exampledata/finetune/bbbp.npz \
               --checkpoint_dir ./model \
               --no_features_scaling \
               --output data_pre.csv
```

### Generating Fingerprints
The pretrained model can also be used to generate the molecular fingerprints. 

**Note**: We provide three ways to generate the fingerprint. 

 - `atom`: The mean pooling of atom embedding from node-view GTransformer and edge-view `GTransformer`. 
 - `bond`: The mean pooling of bond embedding from node-view GTransformer and edge-view `GTransformer`. 
 - `both`: The concatenation of `atom` and `bond` fingerprints. Moreover, the additional molecular features are appended to the output of `GTransformer` as well if provided. 
``` bash
python main.py fingerprint --data_path exampledata/finetune/bbbp.csv \
                           --features_path exampledata/finetune/bbbp.npz \
                           --checkpoint_path model/tryout/model.ep3 \
                           --fingerprint_source both \
                           --output model/fingerprint/fp.npz
```

## The Results

- The classification datasets.

Model      | BBBP | SIDER | ClinTox | BACE | Tox21 | ToxCast 
----       | ---|----       |----       |----        |----       |----       
GROVER<sub>base</sub> |  0.936(0.008) | 0.656(0.006) | 0.925(0.013) | 0.878(0.016) | 0.819(0.020) | 0.723(0.010)
GROVER<sub>large</sub> | 0.940(0.019) | 0.658(0.023) | 0.944(0.021) | 0.894(0.028) | 0.831(0.025) | 0.737(0.010)

- The regression datasets.

Model | FreeSolv | ESOL | Lipo | QM7 | QM8 
----- | ---- | ---- | ---- | ---- | --- 
GROVER<sub>base</sub>  | 1.592(0.072) | 0.888(0.116) |  0.563(0.030) | 72.5(5.9) | 0.0172(0.002) 
GROVER<sub>large</sub> | 1.544(0.397) | 0.831(0.120) |  0.560(0.035) | 72.6(3.8) | 0.0125(0.002)  





## The Reproducibility Issue

Due to the non-deterministic behavior of the function `index_select_nd`( See [link](https://pytorch.org/docs/stable/notes/randomness.html)), it is hard to exactly reproduce the training process of finetuning. Therefore, we provide the finetuned model for eleven datasets to guarantee the reproducibility of the experiments.  
- BBBP:  [BASE](https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/grover_base_ft_refine/bbbp.tar.gz), [LARGE](https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/grover_large_ft_refine/bbbp.tar.gz)
- SIDER: [BASE](https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/grover_base_ft_refine/sider.tar.gz), [LARGE](https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/grover_large_ft_refine/sider.tar.gz)
- ClinTox: [BASE](https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/grover_base_ft_refine/clintox.tar.gz), [LARGE](https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/grover_large_ft_refine/clintox.tar.gz)
- BACE: [BASE](https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/grover_base_ft_refine/bace.tar.gz), [LARGE](https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/grover_large_ft_refine/bace.tar.gz)
- Tox21: [BASE](https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/grover_base_ft_refine/tox21.tar.gz), [LARGE](https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/grover_large_ft_refine/tox21.tar.gz)
- ToxCast: [BASE](https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/grover_base_ft_refine/toxcast.tar.gz), [LARGE](https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/grover_large_ft_refine/toxcast.tar.gz)
- FreeSolv: [BASE](https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/grover_base_ft_refine/freesolv.tar.gz), [LARGE](https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/grover_large_ft_refine/freesolv.tar.gz)
- ESOL: [BASE](https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/grover_base_ft_refine/esol.tar.gz), [LARGE](https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/grover_large_ft_refine/esol.tar.gz)
- Lipo: [BASE](https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/grover_base_ft_refine/lipo.tar.gz), [LARGE](https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/grover_large_ft_refine/lipo.tar.gz)
- QM7: [BASE](https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/grover_base_ft_refine/qm7.tar.gz), [LARGE](https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/grover_large_ft_refine/qm7.tar.gz)
- QM8 [BASE](https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/grover_base_ft_refine/qm8.tar.gz), [LARGE](https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/grover_large_ft_refine/qm8.tar.gz)

We provide the `eval` function to reproduce the experiments. Suppose the finetuned model is placed in `model/finetune/`. 

```
python main.py eval --data_path exampledata/finetune/bbbp.csv \
                    --features_path exampledata/finetune/bbbp.npz \
                    --checkpoint_dir model/finetune/bbbp \
                    --dataset_type classification \
                    --split_type scaffold_balanced \
                    --ensemble_size 1 \
                    --num_folds 3 \
                    --metric auc \
                    --no_features_scaling
```

**Note:** The defualt `metric` setting is `rmse` for regression tasks. 
For `QM7` and `QM8` datasets, you need to set `metric` as  `mae` to reproduce the results. 
For classification tasks, you need to set `metric` as `auc`.

## Known Issues

-  Comparing with the original implementation, we add the `dense` connection in MessagePassing layer in `GTransformer` . If you do not want to add the dense connection in MessagePasssing layer, please fix it at `L256` of `model/layers.py`.

## Roadmap

 - Implementation of `GTransformer` in [DGL](https://github.com/dmlc/dgl) / [PyG](https://github.com/rusty1s/pytorch_geometric).
 - The improvement of self-supervised tasks, e.g. more semantic motifs.

## Reference
```
@article{rong2020self,
  title={Self-Supervised Graph Transformer on Large-Scale Molecular Data},
  author={Rong, Yu and Bian, Yatao and Xu, Tingyang and Xie, Weiyang and Wei, Ying and Huang, Wenbing and Huang, Junzhou},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
## Disclaimer 
This is not an officially supported Tencent product.
