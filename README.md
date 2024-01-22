# SLOmet-ATOMIC 2020: On Symbolic and Neural Commonsense Knowledge Graphs in Slovenian Language

The SloMET-ATOMIC 2020 is a Slovene commonsense reasoning model adapted from the original
COMET-ATOMIC 2020 model.

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

The data set used in these experiments is the [SloATOMIC 2020][sloatomic-data]
data set, a slovene translation of the [ATOMIC 2020][atomic-2020] data set. The
data set has been automatically translated using the [DeepL translation][deepl]
service. In addition, 10k examples have been manually corrected.

## ⚗️ Experiments

To run the experiments, run the folowing commands:

```bash
# retrieve the sloatomic 2020 data set and extract it in the ./data folder
curl --remote-name-all https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1724{/sloatomic2020.zip} && unzip -o sloatomic2020.zip -d ./data && rm -f sloatomic2020.zip

# model training script
python scripts/train_comet_gpt2.py \
    --train_data_path=./data/sloatomic_train.tsv \
    --valid_data_path=./data/sloatomic_dev.tsv \
    --models_dir_path=./models

# model testing script
python scripts/test_comet_gpt2.py \
    --test_data_path=./data/sloatomic_test.tsv.automatic_all \
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
any dependencies into consideration. **NOTE:** The experiment will run the evaluation on
the _manual 10k_ test set (see Results). If an evaluation on a different test set
is required, please change the `test_data_path` parameter and the appropriate
dependencies in the `dvc.yaml` file.

### Results

The results folder contain the files for both evaluating the generations and the
evalution results. File `results/pred_generations_gens_scores.jsonl` show the
performance of the model based on various metrics.

The table below shows the performances of the commonsense models trained using the
corresponding language model and the SloATOMIC dataset. The evaluation was performed
in various ways:

- **Automatic all.** The evaluation performed on the whole automatically translated
  SloATOMIC test set.
- **Automatic 10k.** The evaluation performed on a selection of 10k examples from
  the automatically translated SloATOMIC test set.
- **Manual 10k.** The evaluation performed on the manually inspected and fixed
  examples from the _automatic 10k_ subset.

| Language Model          | Model Lang | Test Set      | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr | METEOR | ROUGE-L |
| ----------------------- | ---------- | ------------- | :----: | :----: | :----: | :----: | :---: | :----: | :-----: |
| Milos/slovak-gpt-j-162M | Slovak     | Automatic all | 0.158  | 0.068  | 0.033  | 0.020  | 0.236 | 0.106  |  0.194  |
|                         |            | Automatic 10k | 0.321  | 0.157  | 0.079  | 0.031  | 0.541 | 0.217  |  0.406  |
|                         |            | Manual 10k    | 0.273  | 0.121  | 0.068  | 0.028  | 0.504 | 0.189  |  0.354  |
| EleutherAI/gpt-neo-125M | English    | Automatic all | 0.263  | 0.128  | 0.051  | 0.023  | 0.431 | 0.184  |  0.351  |
|                         |            | Automatic 10k | 0.256  | 0.069  | 0.000  | 0.000  | 0.498 | 0.178  |  0.355  |
|                         |            | Manual 10k    | 0.221  | 0.046  | 0.001  | 0.000  | 0.472 | 0.160  |  0.320  |
| macedonizer/sl-gpt2     | Slovene    | Automatic all | 0.322  | 0.163  | 0.087  | 0.052  | 0.476 | 0.220  |  0.395  |
|                         |            | Automatic 10k | 0.327  | 0.169  | 0.099  | 0.064  | 0.535 | 0.219  |  0.408  |
|                         |            | Manual 10k    | 0.266  | 0.099  | 0.047  | 0.023  | 0.488 | 0.183  |  0.350  |
| gpt-janezek             | Slovene    | Automatic all | 0.301  | 0.148  | 0.082  | 0.053  | 0.490 | 0.210  |  0.388  |
|                         |            | Automatic 10k | 0.331  | 0.151  | 0.078  | 0.047  | 0.553 | 0.214  |  0.402  |
|                         |            | Manual 10k    | 0.266  | 0.094  | 0.041  | 0.020  | 0.488 | 0.181  |  0.345  |
| gpt-janez               | Slovene    | Automatic all | 0.324  | 0.174  | 0.108  | 0.076  | 0.508 | 0.225  |  0.397  |
|                         |            | Automatic 10k | 0.340  | 0.175  | 0.101  | 0.063  | 0.551 | 0.229  |  0.420  |
|                         |            | Manual 10k    | 0.283  | 0.136  | 0.089  | 0.061  | 0.505 | 0.194  |  0.358  |
| cjvt/gpt-sl-base        | Slovene    | Automatic all | 0.338  | 0.192  | 0.121  | 0.086  | 0.526 | 0.235  |  0.413  |
|                         |            | Automatic 10k | 0.345  | 0.193  | 0.128  | 0.092  | 0.568 | 0.230  |  0.423  |
|                         |            | Manual 10k    | 0.282  | 0.149  | 0.095  | 0.000  | 0.509 | 0.193  |  0.357  |
| COMET(GPT2-XL)          | English    | Original ENG  | 0.407  | 0.248  | 0.171  | 0.124  | 0.653 | 0.292  |  0.485  |

## 🚀 Using the trained model

When the model is trained, use the scripts below to load the model and tokenizer:

```python
# Importing the GPT2 modules from huggingface/transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# define the directory path that contains the model data
MODEL_DIR_PATH = "./models/checkpoint_latest"

# initialize the model and tokenizer with the trained data
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR_PATH)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR_PATH)
```

An example of how it can be used is found in the [scripts/apply_comet_gpt2.py](scripts/apply_comet_gpt2.py) file.

## 📦️ Available models

This project made the following commonsense reasoning model available:

- [SloMET-ATOMIC-2020][sloatomic-model]

## 📚 Papers

If using this code, please cite the following papers:

**[SLOmet - Slovenian Commonsense Description.][published-paper]**
Adrian Mladenić Grobelnik, Erik Novak, Dunja Mladenić, Marko Grobelnik
SiKDD Slovenian KDD Conference, 2022.

### 📓 Related Work

**[(Comet-) Atomic 2020: On Symbolic and Neural Commonsense Knowledge Graphs.][official-comet-atomic]**
Jena D. Hwang, Chandra Bhagavatula, Ronan Le Bras, Jeff Da, Keisuke Sakaguchi, Antoine Bosselut, Yejin Choi
AAAI Conference on Artificial Intelligence, 2021.

## 📣 Acknowledgments

This work is developed by [Department of Artificial Intelligence][ailab] at [Jozef Stefan Institute][ijs].

The work is supported by the Slovenian Research Agency and the [RSDO][rsdo] project.

[python]: https://www.python.org/
[conda]: https://www.anaconda.com/
[git]: https://git-scm.com/
[dvc]: https://dvc.org/
[sloatomic-data]: http://hdl.handle.net/11356/1724
[atomic-2020]: https://allenai.org/data/atomic-2020
[deepl]: https://www.deepl.com/translator
[sl-gpt2]: https://huggingface.co/macedonizer/sl-gpt2
[sloatomic-model]: http://hdl.handle.net/11356/1729
[published-paper]: https://ailab.ijs.si/dunja/SiKDD2022/Papers/SiKDD2022_paper_5674.pdf
[official-comet-atomic]: https://www.semanticscholar.org/paper/COMET-ATOMIC-2020%3A-On-Symbolic-and-Neural-Knowledge-Hwang-Bhagavatula/e39503e01ebb108c6773948a24ca798cd444eb62
[ailab]: http://ailab.ijs.si/
[ijs]: https://www.ijs.si/
[rsdo]: https://www.cjvt.si/rsdo/en/project/
