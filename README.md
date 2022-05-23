# SLOmet-ATOMIC 2020: On Symbolic and Neural Commonsense Knowledge Graphs in Slovenian Language

This project contains the Comet-atomic 2020 model source code modified for the Slovenian language.

## ☑️ Requirements

Before starting the project make sure these requirements are available:

- [python][python]. For executing the code in this project.
- [git][git]. For versioning your code.
- [dvc][dvc]. For versioning your data (part of project requirements).

## 🛠️ Setup

### Create a python environment

First create the virtual environment where the service will store all the modules.

#### Using virtualenv

Using the `virtualenv` command, run the following commands:

```bash
# install the virtual env command
pip install virtualenv

# create a new virtual environment
virtualenv -p python ./.venv

# activate the environment (UNIX)
./.venv/bin/activate

# activate the environment (WINDOWS)
./.venv/Scripts/activate

# deactivate the environment (UNIX & WINDOWS)
deactivate
```

#### Using conda

Install [conda][conda], a program for creating python virtual environments. Then run the following commands:

```bash
# create a new virtual environment
conda create --name slomet2020 python=3.8 pip

# activate the environment
conda activate slomet2020

# deactivate the environment
deactivate
```

### Install

To install the requirements run:

```bash
pip install -e .
```

## 🗃️ Data

To get the data reach out to the project's maintainer.

**NOTE:** The data will be made publicly available. Stay tuned for more!

## ⚗️ Experiments

To run the experiments, run the folowing commands:

```bash
# model training script
python scripts/train_comet_gpt2.py \
    --train_data_path=./data/atomic_train.tsv \
    --valid_data_path=./data/atomic_dev.tsv \
    --models_dir_path=./models

# model testing script
python scripts/test_comet_gpt2.py \
    --test_data_path=./data/atomic_test.tsv \
    --models_dir_path=./models/checkpoint_latest \
    --results_dir_path=./results

# model evaluation script
python scripts/eval_comet_gpt2.py \
    --pred_file_path=./results/pred_generations.jsonl
```

### 🦉 Using DVC

An alternative way of running the whole experiment is by using [DVC][dvc]. To do this,
simply run:

```bash
dvc exp run
```

This command will read the `dvc.yaml` file and execute the stages accordingly, taking
any dependencies into consideration.

### Results

The results folder contain the files for both evaluating the generations and the
evalution results. File `results/pred_generations_gens_scores.jsonl` show the
performance of the model based on various metrics.

## 📦️ Integrated models

This project support the following models:

- [macedonizer/sl-gpt2][sl-gpt2]

## 🚀 Using the trained model

When the model is trained, use the scripts below to load the model and tokenizer:

```python
# Importing the GPT2 modules from huggingface/transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# define the directory path that contains the model data
MODEL_DIR_PATH = "./models/checkpoint_latest"

# initialize the model and tokenizer with the trained data
model = GPT2LMHeadModel.from_pretrained(MODEL_DATA_PATH)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DATA_PATH)
```

## 📚 Papers

TODO

### 📓 Related Work

**[(Comet-) Atomic 2020: On Symbolic and Neural Commonsense Knowledge Graphs.][official-comet-atomic]**\
Jena D. Hwang, Chandra Bhagavatula, Ronan Le Bras, Jeff Da, Keisuke Sakaguchi, Antoine Bosselut, Yejin Choi \
AAAI Conference on Artificial Intelligence, 2021

## 🚧 Work In Progress

- [x] Setup script
- [x] Folder structure
- [x] Code for model training
- [x] Code for model prediction
- [x] Code for model evaluation
- [x] Add support for 3rd party models (outside huggingface)
- [x] Add `params.yaml` and modify the scripts to read the params from the file
- [x] Add DVC pipelines for model training and evaluation
- [ ] Add scripts for storing and retrieving the data set

## 📣 Acknowledgments

This work is developed by [Department of Artificial Intelligence][ailab] at [Jozef Stefan Institute][ijs].

The work is supported by the Slovenian Research Agency and the [RSDO][rsdo] project.

[python]: https://www.python.org/
[conda]: https://www.anaconda.com/
[git]: https://git-scm.com/
[dvc]: https://dvc.org/
[sl-gpt2]: https://huggingface.co/macedonizer/sl-gpt2
[official-comet-atomic]: https://www.semanticscholar.org/paper/COMET-ATOMIC-2020%3A-On-Symbolic-and-Neural-Knowledge-Hwang-Bhagavatula/e39503e01ebb108c6773948a24ca798cd444eb62
[ailab]: http://ailab.ijs.si/
[ijs]: https://www.ijs.si/
[rsdo]: https://www.cjvt.si/rsdo/en/project/
