from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--dataset', type=str, default='gowalla')
parser.add_argument('--dataset_id', type=int, default=0)
parser.add_argument('--strategy', type=str, default='node dropout')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

print(f"\n\nSTARTING TRAINING ON DATASET WITH STRATEGY: {args.strategy} AND ID: {args.dataset_id}...")
run_experiment(f"config_files/{args.dataset}.yml", sampling=args.strategy, idx=args.dataset_id, gpu=args.gpu)
print(f"\n\nTRAINING ENDED")
