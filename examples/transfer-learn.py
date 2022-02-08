import numpy as np
from skopt import Optimizer
import pandas as pd
from sdv.tabular import GaussianCopula
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sdv.tabular import TVAE
from sdv.evaluation import evaluate

import sys

def f(x):
    res = 0
    for i in range(len(x)):
        res = res + (np.sin(5 * x[i]) * (1 - np.tanh(x[i] ** 2)) + np.random.randn() * 0.1)
    return res


df = pd.read_csv('skopt_result.csv',header=0)
print(df)
q_10 = np.quantile(df.objective.values, 0.10)
req_df = df.loc[df['objective'] < q_10]
print(req_df.shape)
req_df = req_df.drop(columns=['objective'])
print(req_df.shape)


space  = [Integer(1, 20, name='epochs'),
          #Integer(1, np.floor(req_df.shape[0]/10), name='batch_size'),
          Integer(1, 8, name='embedding_dim'),
          Integer(1, 8, name= 'compress_dims'),
          Integer(1, 8, name= 'decompress_dims'),
          Real(10**-8, 10**-4, "log-uniform", name='l2scale'),
          Integer(1, 5, name= 'loss_factor')
         ]

@use_named_args(space)
def objective(**params):
    params['epochs'] = 10*params['epochs']
    #params['batch_size'] = 10*params['batch_size']
    params['embedding_dim'] = 2**params['embedding_dim']
    params['compress_dims'] = [2**params['compress_dims'],2**params['compress_dims']]
    params['decompress_dims'] = [2**params['decompress_dims'],2**params['decompress_dims']]
    print(params)
    model = TVAE(**params)
    model.fit(req_df)
    synthetic_data = model.sample(100)
    score = evaluate(synthetic_data, req_df)
    print(score)
    return -score


@use_named_args(space)
def model_fit(**params):
    params['epochs'] = 10*params['epochs']
    #params['batch_size'] = 10*params['batch_size']
    params['embedding_dim'] = 2**params['embedding_dim']
    params['compress_dims'] = [2**params['compress_dims'],2**params['compress_dims']]
    params['decompress_dims'] = [2**params['decompress_dims'],2**params['decompress_dims']]
    print(params)
    model = TVAE(**params)
    model.fit(req_df)
    synthetic_data = model.sample(100)
    score = evaluate(synthetic_data, req_df)
    print(score)
    return -score, model


opt = Optimizer(space, tl_sdv=None)
for i in range(30):
    suggested = opt.ask()
    y = objective(suggested)
    opt.tell(suggested, y)
    print('iteration:', i, suggested, y)

print(opt.yi)

min_value  = min(opt.yi)
min_index = opt.yi.index(min_value)
print(min_value)
best_params = opt.Xi[min_index]

score, model = model_fit(best_params)
print(score)

opt = Optimizer([(-3.0, 3.0),(-3.0, 3.0),(-3.0, 3.0),(-3.0, 3.0),(-3.0, 3.0)], tl_sdv = model)

if 1:
    for i in range(100):
        suggested = opt.ask()
        y = f(suggested)
        opt.tell(suggested, y)
        print('iteration:', i, y)

    df = pd.DataFrame(opt.Xi)
    print(df)
    print(opt.yi)
    df['yi'] = opt.yi
    print(df)
    df.to_csv('tl-skopt_result.csv',index=False)


