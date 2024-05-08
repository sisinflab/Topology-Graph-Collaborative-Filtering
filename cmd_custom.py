#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import pandas as pd
import datetime

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Baby', help='choose the dataset')
parser.add_argument('--gpu_id', type=int, default=0, help='choose the gpu id')
parser.add_argument('--batch_size_jobs', type=int, default=5, help='batch size for jobs')
parser.add_argument('--cluster', type=str, default='cineca', help='cluster name')

args = parser.parse_args()


def to_logfile(row):
    outfile = f"{row['strategy'].replace(' ', '-')}_{row['dataset_id']}.log"
    return outfile


def main():
    logs_path = 'logs'
    scripts_path = 'scripts'

    if not os.path.exists(logs_path + f'/{args.dataset}'):
        os.makedirs(logs_path + f'/{args.dataset}')

    if not os.path.exists(scripts_path + f'/{args.dataset}'):
        os.makedirs(scripts_path + f'/{args.dataset}')

    command_lines = set()

    sampling_stats = pd.read_csv(f'./data/{args.dataset}/sampling-stats.tsv', sep='\t')

    for idx, row in sampling_stats.iterrows():
        strategy = '-'.join(row['strategy'].split(' '))
        dataset_id = row['dataset_id']
        logfile = to_logfile(row)
        completed = False
        if os.path.isfile(f'{logs_path}/{args.dataset}/{logfile}'):
            with open(f'{logs_path}/{args.dataset}/{logfile}', 'r', encoding='utf-8',
                      errors='ignore') as f:
                content = f.read()
                completed = 'TRAINING ENDED' in content

        if not completed:
            command_line = (
                f'python start_experiments_cluster.py '
                f'--dataset={args.dataset} '
                f'--dataset_id={dataset_id} '
                f'--strategy={strategy} '
                f'--gpu={args.gpu_id} > {logs_path}/{args.dataset}/{logfile} 2>&1')
            command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)

    import random
    rng = random.Random(0)
    rng.shuffle(sorted_command_lines)

    nb_jobs = len(sorted_command_lines)

    if args.batch_size_jobs == -1:
        args.batch_size_jobs = nb_jobs

    if args.cluster == 'cineca':
        header = """#!/bin/bash -l

#SBATCH --output=/home/%u/slogs/topology-%A_%a.out
#SBATCH --error=/home/%u/slogs/topology-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --job-name=topology
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB # memory in Mb
#SBATCH --cpus-per-task=4 # number of cpus to use - there are 32 on each node.
#SBATCH --time=4:00:00 # time requested in days-hours:minutes:seconds
#SBATCH --array=1-{0}

echo "Setting up bash environment"
source ~/.bashrc
set -x

# Modules
echo "Setting up modules"
module purge
module load anaconda3/2022.10/gcc-11.2.0
module load cuda/11.8.0/gcc-11.2.0

# Conda environment
echo "Activate conda environment"
source activate topology

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd $HOME/workspace/Topology-Graph-Collaborative-Filtering

echo "Run experiments"
"""
    else:
        header = None

    date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

    if header:
        for index, offset in enumerate(range(0, nb_jobs, args.batch_size_jobs), 1):
            offset_stop = min(offset + args.batch_size_jobs, nb_jobs)
            with open(scripts_path + f'/{args.dataset}/{args.model}/' + date_time + f'__{index}.sh', 'w') as f:
                print(header.format(offset_stop - offset), file=f)
                current_command_lines = sorted_command_lines[offset: offset_stop]
                for job_id, command_line in enumerate(current_command_lines, 1):
                    print(f'test $SLURM_ARRAY_TASK_ID -eq {job_id} && sleep 10 && {command_line}', file=f)


if __name__ == '__main__':
    main()
