# EduOptimNeurostim
GP-BO optimization of neurostimulation. 
Educational material.

Companion code for the following paper : [Link to paper available upon publication]

Please cite the present code as:
Bonizzato M.<sup>&</sup>, Macar U.<sup>&</sup>, Guay-Hottin R., ChoiniËre L., Lajoie G., Dancause N. 2022. "EduOptimNeurostim"ù, GitHub.  https://github.com/mbonizzato/EduOptimNeurostim/. \
<sup>&</sup>, these authors equally contributed to the repository concept and material.


## Usage notes

parameters ([`config`](/config) folder). We built a series of tutorials to help navigate all the available features and design library calls ([`tutorials`](/tutorials) folder).
A few utilities for data downloading and result visualization complement this release ([`scripts`](/scripts) folder).

You may want to run our Tutorial 1 first.

### Getting started


1. Make sure you are equipped to read an IPYNB file, a notebook document created by 
Jupyter Notebook. https://jupyter.org/install

2. The notebooks contain `!bash` commands to download the dataset. If you wish to use 
these, you will need a Git Bash. https://gitforwindows.org/ If you prefer to avoid 
installing this, you can manually download the datasets (see Data section below). 
The bash routine would download them in a `data/` folder, with `data/rat/` and 
`/data/nhp/` subfolders, we suggest to maintain this structure to minimize changes 
to the notebook code.

3. Clone the repository.

```git clone https://github.com/mbonizzato/EduOptimNeurostim.git```

4. You will need [`python 3.7.4`](https://www.python.org/downloads/release/python-374/) to 
run the code. We suggest to create a new virtual environment with this version, e.g.,:

```virtualenv --python=python3.7.4 <PATH_TO_ENVIRONMENT>```

5. After activating the environment, install the requirements:

```pip install -r requirements.txt```

6. Open the notebook Experiment 1 in Jupyter Notebook and enjoy the tutorial!

### Command line interface flags

Given below are the CLI flags that one can use with `main.py`. 
For example use cases, you can check out the [`tutorials`](/tutorials) folder.

* `--config` or `-c`: path to the JSON config file (e.g., `config/rat_mapping_2D.json`)
* `--seed` or `-s`: seed for reproducibility (e.g., `42`)
* `--dataset_path` or `-dp`: path to the dataset of replicates (e.g., `data/rat`)
* `--n_muscles` or `-nm`: number of replicates to run the algorithm on, 
leave empty to run on all available (e.g., `4`)
* `--output_path` or `-op`: output path for results and visualizations 
(e.g., `output/rat_mapping_2D`)
* `--hyperparam_path` or `-hp`: specify hyperparameters with either one of 
(a) path to a `.pkl` file (e.g., `output/rat_mapping_2D/best_hyperparams.pkl`) or 
(b) in-line definition (e.g., `"{'kappa': 3.0}"`)
* `--prior_path` or `-pp`: path to the prior object represented by a MATLAB file (e.g., `priors/rat_mapping_2D.mat`)
* `--algorithm` or `-alg`: algorithm to be used, choose one of  `gpbo`, `extensive`, or `greedy`
* `--greedy_init` or `-ginit`: the initial query location at which greedy search should start, 
only relevant algorithm is `greedy`, (e.g., pass `2,1` for `[2,1]`)
* `--max_queries` or `-mq`: maximum amount of queries to run in each repetition, max queries is
set to number of channels if left empty (e.g., `100`)
* `--n_repetitions` or `-nr`: number of repetitions to run the algorithm for (e.g., `30`)
* `--validation` or `-val`: specify to run validation, otherwise script will perform hyperparameter optimization
* `--gpu` or `-gpu`: specify to run the GPU version
* `--show_warnings` or `-sw`: specify to show warnings, can help during debugging

## Data

The dataset can be downloaded at :  https://osf.io/54vhx/
While running the provided tutorial notebooks, the dataset will be automatically downloaded for you.

Extensive dataset explanation is available at the project's Wiki : https://osf.io/54vhx/wiki/home/

Please cite the dataset as:
Bonizzato M., Massai E.<sup>&</sup>, CÙtÈ S.<sup>&</sup>, Quessy S., Martinez M., and Dancause N. 2021. "OptimizeNeurostim"ù OSF. osf.io/54vhx. \
<sup>&</sup>, these authors equally contributed to the dataset.






