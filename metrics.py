import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Metrics(object):
    """
    Class for storing all kinds of performance estimations
    :param (manager.Config) config: configuration object
    :param (tuple) shape: tuple w/ shape (n_hyperparams, dataset_length, n_runs, n_repetitions, n_queries)
    :param (str) status: either 'online' or 'offline'; decides on how to plot metrics
    """
    def __init__(self, config, shape, status):
        self.config = config
        self.shape = shape
        self.n_hyperparams, self.n_muscles, self.n_runs, self.n_repetitions, self.n_queries = shape
        self.status = status

        # Initialize matrices for storing all metrics at each index
        self.mapping_acc_all = np.empty(shape)
        self.explore_perf_all = np.empty(shape)
        self.exploit_perf_all = np.empty(shape)

        # Initialize arrays for storing aggregated (i.e. mean-value) metrics
        self.explore_perf_agg = None
        self.exploit_perf_agg = None

    def update(self, index, mapping_acc, explore_perf, exploit_perf):
        self.mapping_acc_all[index] = mapping_acc
        self.explore_perf_all[index] = explore_perf
        self.exploit_perf_all[index] = exploit_perf

    def aggregate_and_plot(self, dots=False):
        """
        Function for calculating aggregated metrics and plotting them.

        :param (bool) dots: Set to True to include dots for points in plotting
        """
        def lower_and_upper_bounds(arr):
            """Returns lower and upper bounds"""
            # nm = num. muscles
            # Preprocessing: if nm > 1, take the mean across repetitions for all measurements
            if arr.shape[1] > 1:
                arr = np.mean(arr, axis=2)
            # Preprocessing: if nm = 1, take the single muscle values
            elif arr.shape[1] == 1:
                arr = arr[:, 0, :, :]

            # Compute the mean and std across dataset (nm > 1) or repetitions (nm = 1) dimension
            mean, std = np.mean(arr, axis=1), np.std(arr, axis=1)

            # SEM = standard error measurement
            # Compute SEM: divide std by sqrt of dataset length (nm > 1) or num repetitions (nm = 1)
            sem = std / np.sqrt(arr.shape[1])

            # Compute lower and upper bounds
            lower_bound, upper_bound = mean - sem, mean + sem

            # Get the lower and upper bounds for the last query if offline hyperparam optimization
            if not self.config.validation:
                lower_bound, upper_bound = lower_bound[:, -1], upper_bound[:, -1]

            return np.squeeze(lower_bound), np.squeeze(upper_bound)

        for run_index in range(self.n_runs):
            # Get the performance measures for the current run index
            explore_perf_cur = self.explore_perf_all[:, :, run_index, :, :]
            exploit_perf_cur = self.exploit_perf_all[:, :, run_index, :, :]

            if self.status == 'offline' and not self.config.validation:
                # Aggregate metrics: get last query dim. and average on dataset and repetitions
                self.explore_perf_agg = np.mean(explore_perf_cur, axis=(1, 2))[:, -1]
                self.exploit_perf_agg = np.mean(exploit_perf_cur, axis=(1, 2))[:, -1]

                # Get hyperparam values and keys for plotting
                hyperparam_values = [c[0] for c in self.config.hyperparam_options.hyperparam_combs]
                hyperparam_keys = self.config.hyperparam_options.hyperparam_keys[0]

                # Get the best hyperparam values and save them in a .pkl
                
                if self.config.find_best_wrt == 'exploration':
                    best_hyperparam_index = np.argmax(self.explore_perf_agg)
                elif self.config.find_best_wrt == 'exploitation':
                    best_hyperparam_index = np.argmax(self.exploit_perf_agg)
                best_hyperparam_value = hyperparam_values[best_hyperparam_index]
                if self.n_runs > 1:
                    with open(os.path.join(self.config.output_path, 'best_hyperparams_run=%d.pkl' % run_index), 'wb') as f:
                        pickle.dump({hyperparam_keys: best_hyperparam_value}, f)
                else:
                    with open(os.path.join(self.config.output_path, 'best_hyperparams.pkl'), 'wb') as f:
                        pickle.dump({hyperparam_keys: best_hyperparam_value}, f)

                # Configure log scale for specific set of hyperparameters
                if hyperparam_keys in ['rho_low', 'noise_min', 'noise_max']:
                    hyperparam_values = np.log10(hyperparam_values)

                # Plot hyperparam values vs. aggregated metrics (average on num. data points dim)
                x, x_key = hyperparam_values, 'hyperparam_values'
                y_explore = self.explore_perf_agg
                y_exploit = self.exploit_perf_agg
                plt.plot(x, y_explore, '-ok' if dots else 'k', alpha=0.9, label='Exploration (knowledge of best channel)')
                plt.plot(x, y_exploit, '-ob' if dots else 'b', alpha=0.9, label='Exploitation (stimulation efficacy)')

                # Plot lower/upper confidence bounds & fill region between if num. data points > 1
                explore_perf_lower, explore_perf_upper = lower_and_upper_bounds(explore_perf_cur)
                exploit_perf_lower, exploit_perf_upper = lower_and_upper_bounds(exploit_perf_cur)
                if self.n_muscles > 1 or self.n_repetitions > 1:
                    plt.fill_between(x,
                                     explore_perf_lower,
                                     explore_perf_upper,
                                     color='black',
                                     alpha=0.2)
                    plt.fill_between(x,
                                     exploit_perf_lower,
                                     exploit_perf_upper,
                                     color='blue',
                                     alpha=0.2)

                # Limit the plot (always bw. 0. and 1.) and write y-axis
                plt.ylim([0, 1])
                plt.ylabel('Performance')

                # Write x-axis label depending on whether it has log-scale or not
                if hyperparam_keys in ['rho_low', 'noise_min', 'noise_max']:
                    plt.xlabel('Hyperparameter value (in log scale)')
                else:
                    plt.xlabel('Hyperparameter value')

                # Give title and add legend
                plt.title('%s performance for multiple values of %s' % (self.config.algorithm_name, hyperparam_keys))
                plt.legend(loc='upper left')

                # Save the plot to file (both picture and vector formats)
                if self.n_runs > 1:
                    plt.savefig(os.path.join(self.config.output_path, 'performance_for_%s_run=%d.png' % (hyperparam_keys, run_index)))
                    plt.savefig(os.path.join(self.config.output_path, 'performance_for_%s_run=%d.svg' % (hyperparam_keys, run_index)), format='svg')
                else:
                    plt.savefig(os.path.join(self.config.output_path, 'performance_for_%s.png' % hyperparam_keys))
                    plt.savefig(os.path.join(self.config.output_path, 'performance_for_%s.svg' % hyperparam_keys), format='svg')

            elif self.config.validation:
                # Aggregate metrics by averaging on dataset and repetitions
                self.explore_perf_agg = np.mean(explore_perf_cur, axis=(1, 2))[0, :]
                self.exploit_perf_agg = np.mean(exploit_perf_cur, axis=(1, 2))[0, :]

                # Plot num. queries versus aggregated metrics
                x, x_key = range(1, self.n_queries + 1), 'queries'
                y_explore, y_exploit = self.explore_perf_agg, self.exploit_perf_agg
                plt.plot(x, y_explore, '-ok' if dots else 'k', alpha=0.9, label='Exploration (knowledge of best channel)')
                plt.plot(x, y_exploit, '-ob' if dots else 'b', alpha=0.9, label='Exploitation (stimulation efficacy)')

                # Plot lower/upper confidence bounds & fill region between if num. data points > 1
                explore_perf_lower, explore_perf_upper = lower_and_upper_bounds(explore_perf_cur)
                exploit_perf_lower, exploit_perf_upper = lower_and_upper_bounds(exploit_perf_cur)
                if self.n_muscles > 1 or self.n_repetitions > 1:
                    plt.fill_between(x,
                                     explore_perf_lower,
                                     explore_perf_upper,
                                     color='black',
                                     alpha=0.2)
                    plt.fill_between(x,
                                     exploit_perf_lower,
                                     exploit_perf_upper,
                                     color='blue',
                                     alpha=0.2)

                # Limit the plot (always bw. 0. and 1.) and write x-axis and y-axis
                plt.ylim([0, 1])
                plt.xlabel('Queries')
                plt.ylabel('Performance')

                # Give title and add legend
                plt.title('%s performance vs queries' % self.config.algorithm_name)
                plt.legend(loc='upper left')

                # Save the plot to file (both picture and vector formats)
                if self.n_runs > 1:
                    plt.savefig(os.path.join(self.config.output_path, 'performance_vs_queries_run=%d.png' % run_index))
                    plt.savefig(os.path.join(self.config.output_path, 'performance_vs_queries_run=%d.svg' % run_index), format='svg')
                else:
                    plt.savefig(os.path.join(self.config.output_path, 'performance_vs_queries.png'))
                    plt.savefig(os.path.join(self.config.output_path, 'performance_vs_queries.svg'), format='svg')
            else:
                raise NotImplementedError()

            # Clear plot
            plt.clf()

            # Save the performance as a CSV dataframe
            df = pd.DataFrame({
                x_key: x,
                'exploration': y_explore,
                'exploration_lower': explore_perf_lower,
                'exploration_upper': explore_perf_upper,
                'exploitation': y_exploit,
                'exploitation_lower': exploit_perf_lower,
                'exploitation_upper': exploit_perf_upper,
            })
            df.set_index(x_key, inplace=True)
            if self.n_runs > 1:
                df.to_csv(os.path.join(self.config.output_path, '%s_performance_run=%d.csv' % (self.config.algorithm, run_index)))
            else:
                df.to_csv(os.path.join(self.config.output_path, '%s_performance.csv' % self.config.algorithm))
