import numpy as np
import random
from functools import partial

from deap import base, tools
from deap.tools import initIterate, initRepeat
from toolbox import get_toolbox

random.seed(42)



def register(alias,function,*args,**kargs):
    pfunc = partial(function, *args, **kargs)
    pfunc.__name__ = alias
    pfunc.__doc__ = function.__doc__

    if hasattr(function, "__dict__") and not isinstance(function, type):
        # Some functions don't have a dictionary, in these cases
        # simply don't copy it. Moreover, if the function is actually
        # a class, we do not want to copy the dictionary.
        pfunc.__dict__.update(function.__dict__.copy())

    return pfunc

def initIterate(container, generator):
    return container(generator())

def wrap(population,weights,n):
    return random.choices(population=population,weights=weights,cum_weights=None,k=n)


data = [np.ones((2,2)),np.ones((2,2))]

#toolbox = get_toolbox(data)
#toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=1)
#toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
#toolbox.register("population", tools.initRepeat, list, toolbox.individual)

gen_idx = partial(random.sample, list(range(10)), 10)

population = [0,1,2,3]
weights = [0.0,1,0.0,0.0]
n = 10

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, wrap, population, weights, n)
b = toolbox.individual()
print(b)


diff_idx = partial(wrap, population, weights,n)
print(gen_idx)
print(diff_idx)
result = initIterate(list, diff_idx)
print(result)

result = initIterate(list, gen_idx)
print(type(result))

func = register()
ups = initRepeat(list, initIterate(list, diff_idx) ,n=2)
print(ups)
