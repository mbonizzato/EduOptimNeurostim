import os
import warnings
import itertools
import argparse
from tqdm.autonotebook import tqdm

import numpy as np
import GPy
import torch
import gpytorch
import math

from manager import Config
from metrics import Metrics
from data import get_system
from online import get_online_api
from utils import minmaxnorm, modify_GPy, modify_gpytorch
from algorithms import GP, PriorMeanGPy, optimize, GreedySearch

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default=None, required=True)
parser.add_argument('--seed', '-s', type=int, default=None)
parser.add_argument('--dataset_path', '-dp', type=str, default=None)
parser.add_argument('--n_muscles', '-nm', type=int, default=None)
parser.add_argument('--output_path', '-op', type=str, default=None)
parser.add_argument('--prior_path', '-pp', type=str, default=None)
parser.add_argument('--hyperparam_path', '-hp', type=str, default=None)
parser.add_argument('--algorithm', '-alg', default=None, choices=['gpbo', 'extensive', 'greedy'])
parser.add_argument('--greedy_init', '-ginit', type=str, default=None, help='Pass "2,1" for [2,1]')
parser.add_argument('--max_queries', '-mq', type=int, default=None)
parser.add_argument('--n_repetitions', '-nr', type=int, default=None)
parser.add_argument('--validation', '-val', default=False, action='store_true')
parser.add_argument('--gpu', '-gpu', default=False, action='store_true')
parser.add_argument('--show_warnings', '-sw', default=False, action='store_true', help='Enable for debugging')
parser.add_argument('--step_by_step', '-sbs', default=False)

args = parser.parse_args()

if args.step_by_step and (args.n_muscles!=1 or args.n_repetitions!=1):
    args.n_muscles=1
    args.n_repetitions=1
    print('Step by step demo is only available for n_muscles= 1 and n_repetitions= 1.')
    print('Setting these values to 1 ...')

# Silence run-time warnings unless explicitly specified
if not args.show_warnings:
    warnings.filterwarnings('ignore')

# Modify the GPy package accordingly for better performance
GPy = modify_GPy(GPy_package=GPy)
# Modify the gpytorch package to prevent error
gpytorch = modify_gpytorch(gpytorch_package=gpytorch)

# Initialize config
config = Config(args=args)

# Get the original loaded prior for resetting later if applicable
original_prior = config.prior

# Load system (status): dataset (offline) or input space (online)
system, status = get_system(config=config)

# Get the online API for communicating with the live animal if status is online
if status == 'online':
    online_api = get_online_api(config=config)

# Get maximum queries
max_queries = config.max_queries or system.n_channel

# Initialize metrics object
metrics_shape = (len(config.hyperparam_options), len(system), config.n_runs, config.n_repetitions, max_queries)
metrics = Metrics(config=config, shape=metrics_shape, status=status)

# Store all channel value selections and algorithm predictions in NumPy arrays
channel_values_arr = np.empty(metrics_shape + (system.n_dim, ))
y_mu_arr = np.empty(metrics_shape[:-1] + (system.n_channel, ))


# Store step by step data in NumPy arrays
if config.step_by_step:
    
    sbs_save_query = np.array([1,4,8,12,16,20,24,28,31])
    sbs_arr = np.empty([len(sbs_save_query), 3, system.n_channel])
    sbs_query_idx = np.empty([max_queries,2])
    sbs_best_query = np.empty([max_queries,2])
    

# Iterate over hyperparameters if applicable
for hyperparam_index, hyperparam_option in enumerate(config.hyperparam_options):
    # Iterate over subjects and muscles if applicable
    for muscle_index, muscle in enumerate(tqdm(system, desc='Iterating over muscles')):
        # Process input space for the muscle if applicable, i.e., create param set to sample from
        config.process_input_space()

        # Initialize response array that will be used to used for performance estimations
        R_agg = np.empty(system.n_channel) * np.nan

        if status == 'offline' and not config.toy:
            # Get ground-truth map
            R, R_mean = muscle.get_response(mean=False), muscle.get_response(mean=True)

            # Perform min-max scaling `R` and `R_mean` if applicable
            if config.scaler:
                R = minmaxnorm(R, min_value=config.scaler_min, max_value=config.scaler_max)
                R_mean = minmaxnorm(R_mean, min_value=config.scaler_min, max_value=config.scaler_max)

            # For the offline case, set `R_agg` to `R_mean` by default
            R_agg = R_mean

        for run_index in range(config.n_runs):
            # For the first run of each muscle, reset the prior to the original loaded
            if run_index == 0:
                config.prior = original_prior

            # Apply modifications to the ground-truth if applicable for multi-run setting
            if run_index > 0:
                multiplier = np.random.uniform(config.lower_multiplier, config.upper_multiplier)
                R_agg = R_agg * multiplier

            # Create the kernel and put prior on two lengthscale hyperparameters and variance
            if args.gpu:
                priorbox = gpytorch.priors.SmoothedBoxPrior(a=math.log(config.rho_low),
                                                            b=math.log(config.rho_high),
                                                            sigma=0.01)
                priorbox2 = gpytorch.priors.SmoothedBoxPrior(a=math.log(0.01 ** 2),
                                                             b=math.log(100.0 ** 2),
                                                             sigma=0.01)
                kernel = gpytorch.kernels.MaternKernel(nu=2.5,
                                                       ard_num_dims=system.n_dim,
                                                       lengthscale_prior=priorbox)
                kernel = gpytorch.kernels.ScaleKernel(kernel, outputscale_prior=priorbox2)
                kernel.base_kernel.lengthscale = [1.0] * system.n_dim
                kernel.outputscale = [1.0]

                prior_likelihood = gpytorch.priors.SmoothedBoxPrior(a=config.noise_min ** 2,
                                                                    b=config.noise_max ** 2,
                                                                    sigma=0.01)
                likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=prior_likelihood)
                likelihood.noise = [1.0]
            else:
                kernel = GPy.kern.Matern52(input_dim=system.n_dim,
                                           variance=1.0,
                                           lengthscale=[1.0] * system.n_dim,
                                           ARD=True,
                                           name='Mat52')
                kernel.variance.set_prior(GPy.priors.Uniform(1e-4, 1e4), warning=False)
                kernel.lengthscale.set_prior(GPy.priors.Uniform(config.rho_low, config.rho_high), warning=False)

            # Store all queries in a matrix Q
            # Q[:, :, 0] -> search space positions
            # Q[:, :, 1] -> responses
            Q = np.zeros((config.n_repetitions, max_queries, 2))

            # Iterate over repetitions
            for repetition_index in range(config.n_repetitions):
                # Store maximum response observed, used to normalize responses between 0 and 1
                max_response = 0.0

                # If prior specified use it or else randomly permute the search space
                if config.prior_path:
                    random_queries = np.argsort(config.prior)[::-1]
                else:
                    random_queries = np.random.permutation(system.n_channel)

                # Store best queries in a list Q_best for each repetition
                Q_best = []

                # Initialize kernel hyperparameters (fixed across all experiments)
                if args.gpu:
                    kernel.base_kernel.lengthscale = [1.0] * system.n_dim
                    kernel.outputscale = 1.0
                    likelihood.noise = 1.0  
                else:
                    kernel_variance = 1.0
                    kernel_lengthscale = [1.0] * system.n_dim
                    noise_variance = 1.0

                if config.algorithm == 'greedy':
                    # Initialize greedy search algorithm
                    greedy_search = GreedySearch(ch2xy=system.ch2xy, shape=system.shape)

                    # Set init query location  via (a) `greedy_init` arg, (b) prior, or (c) random
                    if config.greedy_init is None and config.prior_path:
                        greedy_init = system.ch2xy[np.argmax(config.prior)].tolist()
                    elif config.greedy_init is None:
                        greedy_init = system.ch2xy[random_queries[0]].tolist()

                    # Curate next channels to visit in greedy search given the init query location
                    greedy_search.curate_next_channels(channel=greedy_init)

                # Iterate over queries
                for query_index in range(max_queries):
                    # For greedy, optimize one parameter (i.e., dimension/direction) at a time
                    if config.algorithm == 'greedy':
                        if query_index == 0:
                            # Set init query location as the first point to visit in greedy search
                            channel = greedy_init
                        else:
                            # Record response from the previous query
                            greedy_search.record_response(channel=channel, response=response)
                            # Get next channel and find the corresponding query value
                            channel = greedy_search.get_random_channel()

                        # Find the corresponding query (i.e., channel index) for the chosen channel
                        query = np.where((system.ch2xy == channel).all(axis=1))[0][0]

                    # For the specified number of steps or for extensive search, sample randomly
                    elif query_index < config.n_random_steps or config.algorithm == 'extensive':
                        query = random_queries[query_index]
                        
                        if config.step_by_step:
                            sbs_query_idx[query_index] = system.ch2xy[query]
                            
                    # For the rest, acquire next query based on the acquisition function
                    elif config.algorithm == 'gpbo' and config.acquisition == 'ucb':
                        # UCB acquisition
                        acquisition_map = y_mu + config.kappa * np.nan_to_num(np.sqrt(y_var))
                        
                        if config.step_by_step and query_index in sbs_save_query:
                            
                            if query_index==1:
                                 i_sbs= 0
                            elif query_index==31:
                                i_sbs=8   
                            else: 
                                i_sbs= int(query_index/4)
                            
                            sbs_arr[i_sbs,0,:]= y_mu.flatten()*max_response
                            sbs_arr[i_sbs,1,:]= np.nan_to_num(np.sqrt(y_var)).flatten()
                            sbs_arr[i_sbs,2,:]= acquisition_map.flatten()
                            
                                                    
                        # Get next query based on the acquisition map
                        query = np.where(acquisition_map.flatten() == np.max(acquisition_map.flatten()))
                        # Randomly choose a query if there are multiple max values
                        query = query[np.random.randint(len(query))][0]
                        
                        if config.step_by_step:

                            sbs_query_idx[query_index] = system.ch2xy[query]

                    # Update channel value matrix with the current corresponding values (i.e., xy)
                    channel_values_arr[hyperparam_index, muscle_index, run_index, repetition_index, query_index, :] = system.ch2xy[query]

                    # Update the query matrix with search space position
                    Q[repetition_index, query_index, 0] = query

                    # Populate the response scalar based on response type and system status
                    response = None
                    if status == 'offline' and config.response_type == 'all' and not config.toy:
                        # Only get valid responses corresponding to the query and muscle
                        response = R[query]
                        response = response[muscle.sorted_isvalid[query] != 0]
                        # Randomly choose a response from the given list of responses
                        response = np.random.choice(response)
                    elif status == 'offline' and config.response_type == 'mean' and not config.toy:
                        # Get queried response from the mean response
                        response = R_mean[query]
                    elif config.toy:
                        # Compute response as a product of channel values for toy problems
                        response = np.prod(system.ch2xy[query])
                        # Scale response using `h_min` and `h_max` (set to 0 and 1 for no effect)
                        h_min = np.random.normal(config.h_min_mu, config.h_min_sigma)
                        h_max = np.random.normal(config.h_max_mu, config.h_max_sigma)
                        response = response * (h_max - h_min) + h_min
                        # Update response matrix with the ground-truth for toy dataset cases
                        R_agg[query] = response
                    elif status == 'online':
                        # Send channel values to system and get response back for the online setting
                        response = muscle.get_response_from_query(values=system.ch2xy[query], online_api=online_api)

                    # Apply inertia if applicable
                    if config.inertia:
                        previous_response = config.inertia_init if query_index == 0 else Q[repetition_index, query_index - 1, 1]
                        response = response * config.inertia_alpha + previous_response * (1 - config.inertia_alpha)

                    # Apply normal noise if applicable
                    if config.normal_noise:
                        if config.normal_noise_sigma_percent:
                            response += np.random.normal(config.normal_noise_mu, response * config.normal_noise_sigma)
                        else:
                            response += np.random.normal(config.normal_noise_mu, config.normal_noise_sigma)

                    # Apply lower and upper bounds if applicable
                    if config.lower_bound:
                        response = max(config.lower_bound, response)
                    if config.upper_bound:
                        response = min(config.upper_bound, response)

                    # Update the query matrix
                    Q[repetition_index, query_index, 1] = response

                    # Update maximum response observed
                    max_response = max([response, max_response])

                    if config.algorithm == 'gpbo':
                        # Get input observations and observed values for regression
                        X = system.ch2xy[Q[repetition_index, :query_index + 1, 0].astype(int)]
                        Y = Q[repetition_index, :query_index + 1, 1] / max_response
                        # Expand last dim for Y to match shapes with X
                        Y = Y[..., None]

                        # Update initial value of parameters
                        if not args.gpu: 
                            kernel.variance = kernel_variance
                            kernel.lengthscale = kernel_lengthscale

                        if query_index == 0:
                            # Configure mean function if applicable
                            mean_function = None
                            if config.prior_path and config.algorithm == 'gpbo':
                                if not args.gpu:
                                    mean_function = GPy.core.Mapping(input_dim=system.n_dim,
                                                                     output_dim=1)
                                    mean_function.f = PriorMeanGPy(prior=config.prior,
                                                                   ch2xy=system.ch2xy,
                                                                   scale=config.prior_scale)
                                    mean_function.gradients_X = lambda a, b: 0
                                    mean_function.update_gradients = lambda a, b: 0

                            # Initialize the model
                            if args.gpu:
                                likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=prior_likelihood)
                                likelihood.noise = [1.0]
                                model = GP(train_x=torch.tensor(X, dtype=torch.float).to(config.device),
                                           train_y=torch.tensor(Y, dtype=torch.float).to(config.device),
                                           likelihood=likelihood,
                                           kernel=kernel,
                                           prior_map=config.prior * config.prior_scale if config.prior_path else None,
                                           system_shape=system.shape)
                                model = model.to(config.device)

                            else:
                                model = GPy.models.GPRegression(X=X,
                                                                Y=Y,
                                                                kernel=kernel,
                                                                normalizer=None,
                                                                noise_var=noise_variance,
                                                                mean_function=mean_function)

                                # Initialize the constraint of the Gaussian noise
                                model.Gaussian_noise.constrain_bounded(config.noise_min ** 2,
                                                                       config.noise_max ** 2,
                                                                       warning=False)
                        else:
                            if args.gpu:
                                model.covariance_module.base_kernel.lengthscale = [model.covariance_module.base_kernel.lengthscale[0][i].item() for i in range(system.n_dim)]
                                model.covariance_module.base_kernel.outputscale = model.covariance_module.outputscale.item()
                                model.likelihood.noise = model.likelihood.noise[0].item()

                            # Fade the prior if applicable
                            if config.prior_path and config.algorithm == 'gpbo':
                                if config.prior_n_queries and query_index > config.prior_n_queries - 1:
                                    model.mean_function = None

                            # Update the training data
                            if args.gpu:
                                model.set_train_data(inputs=torch.tensor(X, dtype=torch.float).to(config.device),
                                                     targets=torch.tensor(Y, dtype=torch.float).to(config.device),
                                                     strict=False)
                            else:
                                model.set_XY(X, Y)
                                model.Gaussian_noise.variance[0] = noise_variance

                        # GP-BO optimization
                        if args.gpu:
                            model.train()
                            likelihood.train()
                            model, likelihood = optimize(model=model,
                                                         likelihood=likelihood,
                                                         training_iter=10,
                                                         train_x=torch.tensor(X, dtype=torch.float).to(config.device),
                                                         train_y=torch.tensor(Y, dtype=torch.float).to(config.device))
                        else:
                            model.optimize(optimizer='scg',
                                           start=None,
                                           messages=False,
                                           max_iters=10,
                                           clear_after_finish=True)

                        # Run test on all search spaces for the subject
                        if args.gpu:
                            model.eval()
                            likelihood.eval()
                            with torch.no_grad():
                                y = likelihood(model(torch.tensor(system.ch2xy, dtype=torch.float).to(config.device)))
                                y_mu, y_var = y.mean.cpu().numpy(), y.variance.cpu().numpy()

                        else:
                            y_mu, y_var = model.predict(Xnew=system.ch2xy,
                                                        full_cov=False,
                                                        Y_metadata=None,
                                                        include_likelihood=True)
                            
                        y_mu_arr[hyperparam_index, muscle_index, run_index, repetition_index, :] = y_mu.flatten()

                        # Update kernel hyperparameters for non-GPU version
                        if not args.gpu:
                            kernel_variance = model.Mat52.variance[0]
                            kernel_lengthscale = list(model.Mat52.lengthscale)
                            noise_variance = model.Gaussian_noise.variance[0]

                        # Update prior to the estimated map from previous run for multi-run setting
                        if config.n_runs > 1 and repetition_index == config.n_repetitions - 1 and query_index == max_queries - 1:
                            config.prior = y_mu.flatten()

                    if config.algorithm == 'gpbo':
                        # Find best query based on only the unique electrodes previously queried
                        queries = np.unique(Q[repetition_index, :query_index + 1, 0].astype(int))
                        # Find the best query based on GP predictions
                        best_query = queries[(y_mu[queries] == np.max(y_mu[queries])).flatten()]
                        # Randomly choose a query if there are multiple max values
                        best_query = np.random.choice(best_query)
                    else:
                        # Find the best query based on the average response
                        buckets = np.empty((query_index + 1, system.n_channel)) * np.nan
                        for q_i, ch in enumerate(Q[repetition_index, :query_index+1, 0].astype(int)):
                            buckets[q_i, ch] = Q[repetition_index, q_i, 1]

                        best_query = np.nanargmax(np.nanmean(buckets, axis=0))

                    # Update best queries list
                    Q_best.append(best_query)                    
                    
                    if config.step_by_step:
                        sbs_best_query[query_index]= system.ch2xy[best_query]

                if status == 'offline':
                    # Get min-max values for normalization
                    min_, max_ = np.nanmin(R_agg), np.nanmax(R_agg)

                    # Estimate explore-exploit perf. compared to best and worst stimulation points
                    explore_perf = (R_agg[Q_best] - min_) / (max_ - min_)
                    exploit_perf = (R_agg[Q[repetition_index, :, 0].astype(int)] - min_) / (max_ - min_)
                    # NOTE: explore_perf and exploit_perf are scalar arrays with length `n_channel`

                    # Calculate model fitting of ground truth value map for GPBO
                    mapping_acc = None
                    if config.algorithm == 'gpbo':
                        correlation_coeffs = np.corrcoef(R_agg, y_mu.flatten())[0, 1]
                        mapping_acc = correlation_coeffs ** 2

                    # Update all performance estimations
                    metrics_index = (hyperparam_index, muscle_index, run_index, repetition_index)
                    metrics.update(index=metrics_index,
                                   mapping_acc=mapping_acc,
                                   explore_perf=explore_perf,
                                   exploit_perf=exploit_perf)

# Average model predictions on repetitions
y_mu_arr = np.mean(y_mu_arr, axis=3)

# Map `y_mu` using `ch2xy` to the actual dimensions for each muscle
y_mu_mapped_arr = np.empty(tuple(y_mu_arr.shape[:-1]) + tuple(system.shape))
for hyperparam_index in range(y_mu_arr.shape[0]):
    for muscle_index in range(y_mu_arr.shape[1]):
        for run_index in range(y_mu_arr.shape[2]):
            y_mu_mapped = np.empty(system.shape)
            for channel_index in range(y_mu_arr.shape[3]):
                dim_values = system.ch2xy[channel_index, :]
                dim_index = tuple(list(set(system.ch2xy[:, i].tolist())).index(v) for i, v in enumerate(dim_values))
                y_mu_mapped[dim_index] = y_mu_arr[hyperparam_index, muscle_index, run_index, channel_index]
            y_mu_mapped_arr[hyperparam_index, muscle_index, run_index, :] = y_mu_mapped

# Post-process the y_mu_mapped to only show valid outputs for valid regions
if system.n_dim == 2:
    for index in itertools.product(*[range(dim) for dim in system.shape]):
        if index not in [tuple([xi - 1 for xi in x]) for x in system.ch2xy.tolist()]:
            y_mu_mapped_arr[(slice(None), slice(None), slice(None)) + index] = np.nan
            
            
if config.step_by_step:
    
    # Map `sbs_arr` using `ch2xy` to the actual dimensions for each muscle
    sbs_mapped_arr = np.empty(tuple(sbs_arr.shape[:-1]) + tuple(system.shape))
    for query_index in range(sbs_arr.shape[0]):
        for cond_index in range(sbs_arr.shape[1]):
            sbs_mapped = np.empty(system.shape)
            for channel_index in range(sbs_arr.shape[2]):
                dim_values = system.ch2xy[channel_index, :]
                dim_index = tuple(list(set(system.ch2xy[:, i].tolist())).index(v) for i, v in enumerate(dim_values))
                sbs_mapped[dim_index] = sbs_arr[query_index, cond_index,channel_index]
            sbs_mapped_arr[query_index, cond_index, :] = sbs_mapped

    # Post-process the sbs_mapped to only show valid outputs for valid regions
    if system.n_dim == 2:
        for index in itertools.product(*[range(dim) for dim in system.shape]):
            if index not in [tuple([xi - 1 for xi in x]) for x in system.ch2xy.tolist()]:
                sbs_mapped_arr[(slice(None), slice(None), slice(None)) + index] = np.nan     
                
    
# Save step by step data

if config.step_by_step:
    np.save(os.path.join(config.output_path, 'sbs_mapped_arr.npy'), sbs_mapped_arr)
    np.save(os.path.join(config.output_path, 'sbs_save_query.npy'), sbs_save_query)
    np.save(os.path.join(config.output_path, 'sbs_query_idx.npy'), sbs_query_idx.astype(int))
    np.save(os.path.join(config.output_path, 'sbs_best_query.npy'), sbs_best_query.astype(int))


# Save key variables, parameters, and results
np.save(os.path.join(config.output_path, 'y_mu_mapped_arr.npy'), y_mu_mapped_arr)
np.save(os.path.join(config.output_path, 'channel_values_arr.npy'), channel_values_arr)

# Log message to user
print('Hooray! Succesfully saved output files to %s.' % config.output_path)

# Aggregate metrics and visualization for the offline setting (i.e., no metrics for online setting)
if status == 'offline':
    metrics.aggregate_and_plot()
