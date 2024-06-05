import pandas as pd
import numpy as np
import argparse
from config import *
import statsmodels.formula.api as sm

np.random.seed(42)

parser = argparse.ArgumentParser(description="Run regression.")
parser.add_argument('--dataset', type=str, default='gowalla')
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--end_id', type=int, default=600)
parser.add_argument('--characteristics', type=str, nargs='+', default=ACCEPTED_CHARACTERISTICS)
args = parser.parse_args()

results = pd.read_csv(f'data/{args.dataset}/characteristics_{args.start_id}_{args.end_id}.tsv', sep='\t')

models = ['LightGCN', 'DGCF', 'SVDGCN']
metrics = ['Recall']
characteristics = args.characteristics
results[characteristics] = results[characteristics].apply(
        lambda x: (x - x.mean()))
msk = np.random.rand(len(results)) < 0.9
test = results[~msk]
train = results[msk]

for metric in metrics:
    models_results = []
    for idx, model in enumerate(models):
        X = train[characteristics]
        y = train[model + '_' + metric]

        formula_str_ml = y.name + ' ~ ' + '+'.join(characteristics)
        model_ml = sm.ols(formula=formula_str_ml, data=train[characteristics+[model + '_' + metric]])
        fitted_ml = model_ml.fit(cov_type='HC1')

        models_results.append({
            'model': model,
            'score': fitted_ml.rsquared,
            'adjusted_score': fitted_ml.rsquared_adj,
            **fitted_ml.params.to_dict(),
            **fitted_ml.pvalues.rename(lambda x: 'p_'+x).to_dict()
        })
        df = pd.DataFrame.from_dict(models_results)
        df.to_csv(f'data/{args.dataset}/regression_{metric.lower()}_{args.start_id}_{args.end_id}.tsv',
                  sep='\t', index=None)
