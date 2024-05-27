# How to Run

## GPU memory needed

All the experiments is able to run on a single graphic card. However, **if you want to get results on ImageNet, the memory on any single graphic card should be larger than 24 GB.** Around 12 GB is enough for other datasets. 


## How to Install
This code is built on top of the toolbox [SEP.Dassl.pytorch] which conducts a little revision on the original Dassl.pytorch. You can prepare the environment as follows:

```
# Create a conda environment
conda create -n dassl python=3.7

# Activate the environment
conda activate dassl

# Install dependencies
pip install -r requirements.txt

# Install torch (version >= 1.7.1) and torchvision
# Please make sure you have installed the gpu version due to the speed.
# For example:
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```

After that, run `pip install -r requirements.txt` under `SEP_main/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated). Then, you are ready to go.

Follow [DATASETS.md](DATASETS.md) to install the datasets.


## Generalization From Base to New Classes

```bash
bash base2new_train.sh
```
