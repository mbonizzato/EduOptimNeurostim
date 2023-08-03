# EduOptimNeurostim
GP-BO optimization of neurostimulation. 
Educational material.
​
Companion code for the following paper : https://www.cell.com/cell-reports-medicine/fulltext/S2666-3791(23)00118-0#secsectitle0255
​
Please cite the present code as:
Bonizzato M.<sup>&</sup>, Macar U.<sup>&</sup>, Guay-Hottin R., Choinière L., Lajoie G., Dancause N. 2022. "EduOptimNeurostim", GitHub.  https://github.com/mbonizzato/EduOptimNeurostim/. \
<sup>&</sup>, these authors equally contributed to the repository concept and material.
​
​
## Usage notes
​
parameters ([`config`](/config) folder). We built a series of tutorials to help navigate all the available features and design library calls ([`tutorials`](/tutorials) folder).
A few utilities for data downloading and result visualization complement this release ([`scripts`](/scripts) folder).
​
You may want to run our Tutorial 1 first.
​
### Getting started
​
​
1. Make sure you are equipped to read an IPYNB file, a notebook document created by 
Jupyter Notebook. https://jupyter.org/install
​
2. The notebooks contain `!bash` commands to download the dataset. If you wish to use 
these, you will need a Git Bash. https://gitforwindows.org/ If you prefer to avoid 
installing this, you can manually download the datasets (see Data section below). 
The bash routine would download them in a `data/` folder, with `data/rat/` and 
`/data/nhp/` subfolders, we suggest to maintain this structure to minimize changes 
to the notebook code.
​
3. Clone the repository.
​
```git clone https://github.com/mbonizzato/EduOptimNeurostim.git```
​
4. You will need [`python 3.7.4`](https://www.python.org/downloads/release/python-374/) to 
run the code. We suggest to create a new virtual environment with this version, e.g.,:
​
```virtualenv --python=python3.7.4 <PATH_TO_ENVIRONMENT>```
​
5. After activating the environment, install the requirements:
​
```pip install -r requirements.txt```
​
6. Open the notebook Experiment 1 in Jupyter Notebook and enjoy the tutorial!
​
### Command line interface flags
​
Given below are the CLI flags that one can use with `main.py`. 
For example use cases, you can check out the [`tutorials`](/tutorials) folder.
​
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
​
## Data
​
The dataset can be downloaded at :  https://osf.io/54vhx/
While running the provided tutorial notebooks, the dataset will be automatically downloaded for you.
​
Extensive dataset explanation is available at the project's Wiki : https://osf.io/54vhx/wiki/home/
​
Please cite the dataset as:
Bonizzato M., Massai E.<sup>&</sup>, Côté S.<sup>&</sup>, Quessy S., Martinez M., and Dancause N. 2021. "OptimizeNeurostim", OSF. osf.io/54vhx. \
<sup>&</sup>, these authors equally contributed to the dataset.
​
​
## Config file variables
​
1. **seed**:
    - Controls: Seeds randomness in the algorithm for reproducibility. 
    - Possible values: Integers
​
2. **output_path**:
   - Controls: Where the algorithm’s outputs are saved in the directory.
​
3. **data**:
    - **dataset_path**:
        - Controls: Specifies where to find the data when running algorithm on pre-collected data.
        - Possible values: A path or null if data is acquired online.
   
   - **selected_muscles**:
        - Controls: Specifies the response of which muscle to optimize. 
        - Possible values: null (muscles are randomly selected) or list of strings. Possible muscle names for rat dataset are "LeftGluteus", "LeftKneeExtensor", "LeftKneeFlexor", "LeftMedialisGastrocnemius", "LeftTibialisAnterioris", "RightGluteus", "RightMedialisGastrocnemius", "RightTibialisAnterioris". Possible muscle names for nhp dataset are "AbductorPollicisBrevis", "AdductorDigitiMinimi", "AdductorPollicis", "ExtensorCarpiRadialis", "ExtensorDigitorumCommunis", "FirstDorsalInterosseous", "FirstDorsalInterosseus", "FlexorCarpiUlnaris", "FlexorDigitorumSuperficialis", "FlexorPollicisBrevis", "OpponensPollicis".
​
   - **online**:
        - Controls: Specifies if the algorithm is to get data online.
        - Possible values: True or False
        - Default value: False
​
   - **online_api**:
        - Controls: Specifies throught which api the online is made available.
        - Possible values: "synapse"

4. **eletrode_mapping_path**:
    - Controls: Specifies where to find a .json file containing the tdt channels and their corresponding coordinate on the physical electrode array. 
    - Possible values: A path to a .json file containing a mapping formated as in config/ch2xy_online.json.

5. **prior**:
    - **path**:
        - Controls: In presence of a prior, specifies where to load it from.
        - Possible values: a path or null
   
    - **scale**:
        - Controls: When loading the prior, specifies by which factor to scale it.
        - Possible values: Floats
​
6. **acquisition**:
    - **name**:
        - Controls: The acquisition function to use to select the next query to execute.
        - Possible values: "ucb"
   
    - **kappa**: Parameter controling the trade-off between exploration and exploitation in the acquisition function UCB.
        - **default**:
            - Controls: The default value of kappa to use when running the algorithm.
            - Possible values: Floats
        - **values**:
            - Controls: The values that will be tested in hyperparameter optimization.
            - Possible values: a list of Floats
        - **find_best**:
            - Controls: If the kappa hyperparameter is optimized when hyperparameter optimization is called.
            - Possible values: True or False
​
7. **optimization**:
    - **name**:
        - Controls: Which algorithm is used in the optimization process.
        - Possible values: "gpbo", "extensive" or "greedy"
​
    - **max_queries**:
        - Controls: The number of queries that will be performed. 
        - Possible values: Positive Integers
​
    - **number_repetitions**:
        - Controls: How many repetitions of the algorithmic run are performed.
        - Possible values: Positive Integers
​
    - **n_random_steps**:
        - **default**:
            - Controls: The default number of queries executed randomly before the acquisition function takes over. 
            - Possible values: Positive integers
        - **values**:
            - Controls: The values that will be tested in hyperparameter optimization.
            - Possible values: List of positive integers
        - **find_best**:
            - Controls: If the number of random steps is optimized when hyperparameter optimization is called.
            - Possible values: True or False
​
8. **noise_min**:
    - **default**:
        - Controls: The default lower bound of the prior over the likelihood's noise parameter.
        - Possible values: Floats
    - **values**:
        - Controls: The values that will be tested in hyperparameter optimization.
        - Possible values: List of floats
    - **find_best**:
        - Controls: If the lower bound of the prior over the likelihood's noise parameter is optimized when hyperparameter optimization is called.
        - Possible values: True or False
​
9. **noise_max**:
    - **default**:
        - Controls: The default higher bound of the prior over the likelihood's noise parameter.
        - Possible values: Floats 
    - **values**:
        - Controls: The values that will be tested in hyperparameter optimization.
        - Possible values: List of floats
    - **find_best**:
        - Controls: If the higher bound of the prior over the likelihood's noise parameter is optimized when hyperparameter optimization is called.
        - Possible values: True or False
​
10. **rho_high**:
    - **default**:
        - Controls: The default higher bound of the prior over kernel lenghtscales.
        - Possible values: Floats
    - **values**:
        - Controls: The values that will be tested in hyperparameter optimization.
        - Possible values: List of floats
    - **find_best**:
        - Controls: If the higher bound of the prior over kernel lenghtscales is optimized when hyperparameter optimization is called.
        - Possible values: True or False
​
11. **rho_low**:
    - **default**:
        - Controls: The default lower bound of the prior on kernel lenghtscales.
        - Possible values: Floats
    - **values**:
        - Controls: The values that will be tested in hyperparameter optimization.
        - Possible values: List of floats
    - **find_best**:
        - Controls: If the lower bound of the prior on kernel lenghtscales is optimized when hyperparameter optimization is called.
        - Possible values: True or False