from deap import base, creator, gp, tools, algorithms
import numpy as np

from toolbox import get_toolbox
from utils import objective

data = [np.ones((2,2)),np.ones((2,2))]

toolbox = get_toolbox(data)

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

pop = toolbox.population(n=300)
hof = tools.HallOfFame(1)
#pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 2, stats=mstats,
#                                   halloffame=hof, verbose=True)

#print("Objective: ", objective(data[0], data[1]))
#print(hof[0])




#func = toolbox.compile(expr=hof[0])
#print(str(func))
#print(weight(hof[0],data))
