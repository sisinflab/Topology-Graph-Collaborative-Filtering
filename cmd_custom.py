#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import pandas as pd
import datetime

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='gowalla', help='choose the dataset')
parser.add_argument('--gpu_id', type=int, default=0, help='choose the gpu id')
parser.add_argument('--batch_size_jobs', type=int, default=5, help='batch size for jobs')
parser.add_argument('--cluster', type=str, default='cineca', help='cluster name')
parser.add_argument('--mail_user', type=str, default='', help='your email')
parser.add_argument('--account', type=str, default='', help='project name')

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
#SBATCH --job-name=SisInf_Topology
#SBATCH --time=24:00:00                                   ## format: HH:MM:SS
#SBATCH --nodes=1
#SBATCH --mem=20GB                                       ## memory per node out of 494000MB (481GB)
#SBATCH --output=$HOME/graph_topology/slogs/SisInf_Topology_output-%A_%a.out
#SBATCH --error=$HOME/graph_topology/slogs/SisInf_Topology_error-%A_%a.err
#SBATCH --account={1}
#SBATCH --mail-type=ALL
#SBATCH --mail-user={2}
#SBATCH --gres=gpu:1                                    ##    1 out of 4 or 8
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --array=1-{0}

module load profile/deeplrn
module load cuda/12.1
module load gcc/12.2.0-cuda-12.1
module load python/3.10.8--gcc--11.3.0

source $HOME/graph_topology/Topology-Graph-Collaborative-Filtering/venv310/bin/activate

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd $HOME/graph_topology/Topology-Graph-Collaborative-Filtering

echo "Run experiments"
"""
    else:
        header = None

    date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

    if header:
        for index, offset in enumerate(range(0, nb_jobs, args.batch_size_jobs), 1):
            offset_stop = min(offset + args.batch_size_jobs, nb_jobs)
            with open(scripts_path + f'/{args.dataset}/' + date_time + f'__{index}.sh', 'w') as f:
                print(header.format(offset_stop - offset, args.account, args.mail_user), file=f)
                current_command_lines = sorted_command_lines[offset: offset_stop]
                for job_id, command_line in enumerate(current_command_lines, 1):
                    print(f'test $SLURM_ARRAY_TASK_ID -eq {job_id} && sleep 10 && {command_line}', file=f)


if __name__ == '__main__':
    main()
