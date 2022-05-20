
# User Preference-aware Graph Neural Network with Attention for Fake News Detection

## Directory structure:

```
.
|   README.md
|   requirement.txt
|
|--- data
|--- model
|--- ouput
|--- src
|   |-- config
|   |   gnn_att_gos.json
|   |   gnn_att_pol.json
|   |   gnn_gos.json
|   |   gnn_pol.json
|   dataset.py
|   main.py
|   model.py
|   trainer.py
|   utils.py
```

## Installation

### Libraries

To install all neccessary libraries, please run:

```bash
conda env create -f environment.yml
```

In case, the version of Pytorch and Cuda are not compatible on your machine, please remove all related lib in the `.yml` file; then install Pytorch and Pytorch Geometric separately.


### PyTorch
Please follow Pytorch installation instruction in this [link](https://pytorch.org/get-started/locally/).


### Torch Geometric
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```
where `${TORCH}` and `${CUDA}` is version of Pytorch and Cuda.


## Model Architecture

![Model architecture](/figure/overview.png)

[Source](https://github.com/safe-graph/GNN-FakeNews)

### Data preparation
The data used in this project in in-memory dataset provided by Torch Geometric. To download the data, run:

```python
train_data = UPFD(root="./data", name="gossipcop", feature="content", split="train")
val_data = UPFD(root="./data", name="gossipcop", feature="content", split="val")
test_data = UPFD(root="./data", name="gossipcop", feature="content", split="test")

train_data = UPFD(root="./data", name="politifact", feature="content", split="train")
val_data = UPFD(root="./data", name="politifact", feature="content", split="val")
test_data = UPFD(root="./data", name="politifact", feature="content", split="test")
```

### Training and Testing

After having the best hyperparameters, edit config files in `./src/config/`

To run training and testing, run:

```bash
sh script/train_test.sh
```

To see the training progress and testing result, check the log files in the folder: `./log`

### Experiment notebook 
To run the experiments in an interactive platform, please check the notebook: `./experiment/main.ipynb`

## Saved checkpoint

Our trained model are saved in the folder: `./model`

## Acknowledgment

This project is done under the supervison of Dr. Kandasamy Illanko at Ryerson University.
