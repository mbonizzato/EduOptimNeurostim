import os
import re
import json
import itertools
import pickle
import numpy as np
from scipy.io import loadmat
from scipy.stats import gamma
import torch

from utils import set_random_seed, minmaxnorm

defined_acquisitions = ['ucb']
defined_algorithms = ['greedy', 'extensive', 'gpbo']


class Config(object):
    def __init__(self, args):
        # Read JSON config file
        assert os.path.exists(args.config) and args.config.endswith('.json')
        with open(args.config) as f:
            config = json.load(f)

        # Set seed for reproducibility
        self.seed = args.seed or config['seed']
        set_random_seed(seed=self.seed)

        # Create output folder for experiments
        self.output_path = args.output_path or config['output_path']
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # Get main parameters for data loading and processing
        self.online = config['data']['online'] if 'online' in config['data'] else None
        self.online_api = config['data']['online_api'] if 'online_api' in config['data'] else None
        self.dataset_path = args.dataset_path or config['data']['dataset_path']
        self.n_muscles = args.n_muscles or (config['data']['n_muscles'] if 'n_muscles' in config['data'] else None)
        self.selected_muscles = config['data']['selected_muscles']
        self.n_repetitions = args.n_repetitions or config['optimization']['n_repetitions']
        self.response_type = 'all' if 'response_type' not in config['data'] else config['data']['response_type']
        self.validation = args.validation or ('validation' in config and config['validation'])
        self.find_best_wrt = 'exploration' if 'find_best_wrt' not in config else config['find_best_wrt']
        self.step_by_step = args.step_by_step

        # Mapping electrodes from TDT channel to electrode coordinates online
        self.mapping_electrodes_online = None if 'eletrode_mapping_path' not in config else config['eletrode_mapping_path']

        # Process input space if applicable
        self.input_space = None if 'input_space' not in config else config['input_space']
        self.process_input_space()

        # Process toy problem details (mainly for experiment 4 as described in email)
        self.toy = False
        if 'toy' in config:
            self.toy = True
            self.toy_n = args.n_muscles or config['toy']['n']
            self.h_min_mu = config['toy']['h_min']['mu']
            self.h_min_sigma = config['toy']['h_min']['sigma']
            self.h_max_mu = config['toy']['h_max']['mu']
            self.h_max_sigma = config['toy']['h_max']['sigma']

        # Get and check main parameters for acquisition and optimization
        self.acquisition = config['acquisition']['name']
        self.algorithm = args.algorithm or config['optimization']['name']
        assert self.algorithm in defined_algorithms and self.acquisition in defined_acquisitions
        self.algorithm_name = {'gpbo': 'GPBO', 'extensive': 'Extensive Search', 'greedy': 'Greedy Search'}[self.algorithm]

        # Get max queries to checkout in the search
        self.max_queries = args.max_queries or (None if 'max_queries' not in config['optimization'] else config['optimization']['max_queries'])

        # Get initial query location for greedy search if applicable
        self.greedy_init = args.greedy_init or (None if 'greedy_init' not in config['optimization'] else config['optimization']['greedy_init'])
        if self.greedy_init:
            # Remove all whitespaces
            self.greedy_init = re.sub(r"\s+", '', self.greedy_init).strip()
            # Convert to a list of integers by splitting on commas
            self.greedy_init = [int(index) for index in self.greedy_init.split(',')]

        # Set hyperparameters to default values
        self.rho_low = config['optimization']['rho_low']['default']
        self.rho_high = config['optimization']['rho_high']['default']
        self.n_random_steps = config['optimization']['n_random_steps']['default']
        self.kappa = config['acquisition']['kappa']['default']
        self.noise_min = config['optimization']['noise_min']['default']
        self.noise_max = config['optimization']['noise_max']['default']

        # Get hyperparameters to tune if applicable
        self.hyperparam_options = [()]
        if not self.validation:
            self.hyperparam_options = HyperparamOptions(config=config, config_object=self)

        # Set hyperparams using the supplied .pkl file or hyperparam string if applicable
        self.hyperparam_path = args.hyperparam_path or (None if 'hyperparam_path' not in config else config['hyperparam_path'])
        if self.hyperparam_path and not self.validation:
            raise ValueError('You can only supply pre-set hyperparameters during validation!')

        if self.hyperparam_path:
            # Read .pkl file as the hyperparam object
            if self.hyperparam_path.endswith('.pkl'):
                with open(self.hyperparam_path, 'rb') as f:
                    hyperparam_object = pickle.load(f)
            # Convert string to dictionary and set it to the hyperparam object
            else:
                hyperparam_object = eval(self.hyperparam_path)

            # Set config attributes by iterating over the keys and values of hyperparam object
            for key, value in hyperparam_object.items():
                setattr(self, key, value)

        # Load preprocessing steps (scaling, normal noise, and inertia) if applicable
        self.scaler = False
        self.normal_noise = False
        self.inertia = False
        self.lower_bound = None
        self.upper_bound = None
        if 'preprocessing' in config and 'scaler' in config['preprocessing']:
            self.scaler = True
            self.scaler_min = config['preprocessing']['scaler']['min']
            self.scaler_max = config['preprocessing']['scaler']['max']
        if 'preprocessing' in config and 'normal_noise' in config['preprocessing']:
            self.normal_noise = True
            self.normal_noise_mu = config['preprocessing']['normal_noise']['mu']
            self.normal_noise_sigma = config['preprocessing']['normal_noise']['sigma']
            self.normal_noise_sigma_percent = False
            if 'sigma_percent' in config['preprocessing']['normal_noise']:
                self.normal_noise_sigma_percent = config['preprocessing']['normal_noise']['sigma_percent']
        if 'preprocessing' in config and 'inertia' in config['preprocessing']:
            self.inertia = True
            self.inertia_init = config['preprocessing']['inertia']['init']
            self.inertia_alpha = config['preprocessing']['inertia']['alpha']
        if 'preprocessing' in config and 'lower_bound' in config['preprocessing']:
            self.lower_bound = config['preprocessing']['lower_bound']
        if 'preprocessing' in config and 'upper_bound' in config['preprocessing']:
            self.upper_bound = config['preprocessing']['upper_bound']

        # Load parameters related to runs (e.g., days)
        self.n_runs = 1 if 'n_runs' not in config['optimization'] else config['optimization']['n_runs']
        self.lower_multiplier = 1.0
        self.upper_multiplier = 1.0
        if 'run_modifications' in config['optimization'] and 'lower_multiplier' in config['optimization']['run_modifications']:
            self.lower_multiplier = config['optimization']['run_modifications']['lower_multiplier']
        if 'run_modifications' in config['optimization'] and 'upper_multiplier' in config['optimization']['run_modifications']:
            self.upper_multiplier = config['optimization']['run_modifications']['upper_multiplier']

        # Load the prior if applicable
        self.prior = None
        self.prior_path = args.prior_path or config['prior']['path']
        self.prior_scale = 1.0 if 'scale' not in config['prior'] else config['prior']['scale']
        self.prior_n_queries = None if 'n_queries' not in config['prior'] else config['prior']['n_queries']
        if self.prior_path is not None:
            assert os.path.exists(self.prior_path) and self.prior_path.endswith('.mat')
            self.prior = loadmat(self.prior_path)['avgMap'].reshape((1, -1))
            # assert self.prior.shape == (1, self.n_channel)
            # Convert prior to a list with length `self.n_channel`
            self.prior = self.prior[0, :]
            # Min-max normalization to 0 and 1
            self.prior = minmaxnorm(self.prior)

        # Configure GPU settings
        if args.gpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
            print('Using device=%s' % self.device)

    def process_input_space(self):
        # If a list of values is not given for any input space dim, generate randomly via gamma PDF
        if self.input_space:
            for param in self.input_space:
                if not isinstance(self.input_space[param], list):
                    # Get all required hyperparameters for random generation
                    a_mu = self.input_space[param]['a']['mu']
                    a_sigma = self.input_space[param]['a']['sigma']
                    a = np.random.normal(a_mu, a_sigma)
                    b = self.input_space[param]['b']
                    low = self.input_space[param]['low']
                    high = self.input_space[param]['high']

                    # Randomly generate values for the current parameter
                    values = gamma.pdf(np.arange(low, high + 1), a=a, scale=b)

                    # Min-max normalization if applicable
                    if 'scaler' in self.input_space[param]:
                        scaler_min = None if 'min' not in self.input_space[param]['scaler'] else \
                            self.input_space[param]['scaler']['min']
                        scaler_max = None if 'max' not in self.input_space[param]['scaler'] else \
                            self.input_space[param]['scaler']['max']

                        if scaler_min and scaler_max:
                            values = (values - values.min()) / (values.max() - values.min())
                            values = values * (scaler_max - scaler_min) + scaler_min
                        elif scaler_min:
                            raise NotImplementedError()
                        elif scaler_max:
                            values = values / values.max()
                            values = values * scaler_max

                    # Update the config with list of generated values
                    self.input_space[param] = values.tolist()
                elif not all([isinstance(value, float) for value in self.input_space[param]]):
                    values = []
                    for value in self.input_space[param]:
                        if isinstance(value, str):
                            mu, sigma = [float(x.strip()) for x in value.split('+/-')]
                            value = np.random.normal(mu, sigma)
                        elif not isinstance(value, float):
                            value = float(value)
                        values.append(value)

                    self.input_space[param] = values


class HyperparamOptions(object):
    """Class for representing a set of hyperparameters"""
    def __init__(self, config, config_object):
        self.config = config
        self.config_object = config_object

        # Create list of values for each hyperparam to-be-tuned
        self.hyperparam = {}

        # For now, we will assume all hyperparameters will be on the second level
        for key in config:
            if isinstance(config[key], dict):
                for subkey in config[key]:
                    if isinstance(config[key][subkey], dict):
                        for subsubkey in config[key][subkey]:
                            if subsubkey == 'find_best':
                                if config[key][subkey][subsubkey] is True:
                                    self.hyperparam[subkey] = config[key][subkey]['values']

        # Aggregate and create all possible value combinations for the given config
        self.hyperparam_keys = list(self.hyperparam.keys())
        self.hyperparam_combs = [c for c in itertools.product(*[v for v in self.hyperparam.values()])]

    def __len__(self):
        return len(self.hyperparam_combs)

    def __getitem__(self, index):
        # Retrieve current hyperparam comb
        hyperparam_comb = self.hyperparam_combs[index]

        # Update hyperparameters in config object according to the current hyperparam combination
        for i, k in enumerate(self.hyperparam_keys):
            setattr(self.config_object, k, hyperparam_comb[i])

        return hyperparam_comb
