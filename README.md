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
conda create --name fastapi-ml python=3.8 pip

# activate the environment
conda activate fastapi-ml

# deactivate the environment
deactivate
```

### Install

To install the requirements run:

```bash
pip install -e .
```

## 📚 Papers

TODO

### 📓 Related Work

**[(Comet-) Atomic 2020: On Symbolic and Neural Commonsense Knowledge Graphs.][official-comet-atomic]**\
Jena D. Hwang, Chandra Bhagavatula, Ronan Le Bras, Jeff Da, Keisuke Sakaguchi, Antoine Bosselut, Yejin Choi \
AAAI Conference on Artificial Intelligence, 2021

## 🚧 Work In Progress

- [x] Setup script
- [ ] Folder structure
- [ ] Code for model training
- [ ] Code for model evaluation
- [ ] Add support for 3rd party models (outside huggingface)
- [ ] Add `params.yaml` and modify the scripts to read the params from the file
- [ ] Add DVC pipelines for model training and evaluation
- [ ] Add scripts for storing and retrieving the data set

## 📣 Acknowledgments

This work is developed by [Department of Artificial Intelligence][ailab] at [Jozef Stefan Institute][ijs].

The work is supported by the Slovenian Research Agency and the [RSDO][rsdo] project.

[python]: https://www.python.org/
[conda]: https://www.anaconda.com/
[git]: https://git-scm.com/
[dvc]: https://dvc.org/
[official-comet-atomic]: https://www.semanticscholar.org/paper/COMET-ATOMIC-2020%3A-On-Symbolic-and-Neural-Knowledge-Hwang-Bhagavatula/e39503e01ebb108c6773948a24ca798cd444eb62
[ailab]: http://ailab.ijs.si/
[ijs]: https://www.ijs.si/
[rsdo]: https://www.cjvt.si/rsdo/en/project/
