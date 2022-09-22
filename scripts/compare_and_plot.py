import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--run1', '-r1', type=str, default=None, required=True)
parser.add_argument('--run2', '-r2', type=str, default=None, required=True)
parser.add_argument('--run3', '-r3', type=str, default=None, required=False)
parser.add_argument('--run_index', '-ri', type=int, default=[-1], nargs="*", required=False, help='Only specify for continual learning experiemnt with multiple days')
parser.add_argument('--dots', '-d', default=False, action='store_true', help='Set to True to include dots for points in plotting')
args = parser.parse_args()

for run_index in args.run_index:
    # Create lists for data frames and performance filename paths for different algorithms
    dfs, fname_ids, runs = [], ['gpbo', 'extensive', 'greedy'], [args.run1, args.run2, args.run3]
    if args.run3 is None:
        fname_ids, runs = fname_ids[:-1], runs[:-1]

    # Populate the previously created lists
    for fname_id in fname_ids:
        for run in runs:
            if args.run_index != [-1]:
                path = os.path.join(run, '%s_performance_run=%d.csv' % (fname_id, run_index))
            else:
                path = os.path.join(run, '%s_performance.csv' % fname_id)
            if os.path.exists(path):
                dfs.append(pd.read_csv(path))
                break

    # EXPLORATION
    plt.plot(dfs[0]['queries'], dfs[0]['exploration'], '-ok' if args.dots else 'k', alpha=0.9, label='GPBO')
    plt.plot(dfs[1]['queries'], dfs[1]['exploration'], '-or' if args.dots else 'r', alpha=0.9, label='Extensive')
    if len(dfs) == 3:
        plt.plot(dfs[2]['queries'], dfs[2]['exploration'], '-o' if args.dots else '-', color='orange', alpha=0.9, label='Greedy')

    # Error bars
    plt.fill_between(dfs[0]['queries'],
                     dfs[0]['exploration_lower'],
                     dfs[0]['exploration_upper'],
                     color='black',
                     alpha=0.2)
    plt.fill_between(dfs[1]['queries'],
                     dfs[1]['exploration_lower'],
                     dfs[1]['exploration_upper'],
                     color='red',
                     alpha=0.2)
    if len(dfs) == 3:
        plt.fill_between(dfs[2]['queries'],
                         dfs[2]['exploration_lower'],
                         dfs[2]['exploration_upper'],
                         color='orange',
                         alpha=0.2)

    # Limit the plot (always bw. 0. and 1.) and write x-axis and y-axis
    plt.ylim([0, 1])
    plt.xlabel('Queries')
    plt.ylabel('Performance')

    # Give title and add legend
    plt.title('Exploration performance vs queries')
    plt.legend(loc='upper left')

    # Save the plot to file (both picture and vector formats)
    if args.run_index != [-1]:
        plt.savefig(os.path.join(args.run1, 'exploration_comparison_run=%d.png' % run_index))
        plt.savefig(os.path.join(args.run1, 'exploration_comparison_run=%d.svg' % run_index), format='svg')
    else:
        plt.savefig(os.path.join(args.run1, 'exploration_comparison.png'))
        plt.savefig(os.path.join(args.run1, 'exploration_comparison.svg'), format='svg')

    # Clear plot
    plt.clf()

    # EXPLOITATION
    plt.plot(dfs[0]['queries'], dfs[0]['exploitation'], '-ok' if args.dots else 'k', alpha=0.9, label='GPBO')
    plt.plot(dfs[1]['queries'], dfs[1]['exploitation'], '-or' if args.dots else 'r', alpha=0.9, label='Extensive')
    if len(dfs) == 3:
        plt.plot(dfs[2]['queries'], dfs[2]['exploitation'], '-o' if args.dots else '-', color='orange', alpha=0.9, label='Greedy')

    # Error bars
    plt.fill_between(dfs[0]['queries'],
                     dfs[0]['exploitation_lower'],
                     dfs[0]['exploitation_upper'],
                     color='black',
                     alpha=0.2)
    plt.fill_between(dfs[1]['queries'],
                     dfs[1]['exploitation_lower'],
                     dfs[1]['exploitation_upper'],
                     color='red',
                     alpha=0.2)
    if len(dfs) == 3:
        plt.fill_between(dfs[2]['queries'],
                         dfs[2]['exploitation_lower'],
                         dfs[2]['exploitation_upper'],
                         color='orange',
                         alpha=0.2)
    # Limit the plot (always bw. 0. and 1.) and write x-axis and y-axis
    plt.ylim([0, 1])
    plt.xlabel('Queries')
    plt.ylabel('Performance')

    # Give title and add legend
    plt.title('Exploitation performance vs queries')
    plt.legend(loc='upper left')

    # Save the plot to file (both picture and vector formats)
    if args.run_index != [-1]:
        plt.savefig(os.path.join(args.run1, 'exploitation_comparison_run=%d.png' % run_index))
        plt.savefig(os.path.join(args.run1, 'exploitation_comparison_run=%d.svg' % run_index), format='svg')
    else:
        plt.savefig(os.path.join(args.run1, 'exploitation_comparison.png'))
        plt.savefig(os.path.join(args.run1, 'exploitation_comparison.svg'), format='svg')

    # Clear plot
    plt.clf()

# Log message to user
print('Hooray! Successfully compared and plotted the runs!')
