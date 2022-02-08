import numpy as np
from skopt import Optimizer
import pandas as pd

def f(x):
    res = 0
    for i in range(len(x)):
        res = res + (np.sin(5 * x[i]) * (1 - np.tanh(x[i] ** 2)) + np.random.randn() * 0.1)
    return res

opt = Optimizer([(-3.0, 3.0),(-3.0, 3.0),(-3.0, 3.0),(-3.0, 3.0),(-3.0, 3.0)])

for i in range(100):
    suggested = opt.ask()
    y = f(suggested)
    opt.tell(suggested, y)
    print('iteration:', i, suggested, y)

df = pd.DataFrame(opt.Xi)
print(df)
print(opt.yi)
df['yi'] = opt.yi
print(df)
df.to_csv('skopt_result.csv',index=False)




