from deap import base, creator, gp, tools
from deap.gp import PrimitiveTree

from primitive_set import pset
from utils import objective, fitness

def get_toolbox(data):
    toolbox = base.Toolbox()
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin,
                   pset=pset)

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=1)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    def weight(individual, data):
        x = data[0]
        y = data[1]

        #tree = PrimitiveTree(individual)
        #tree.
        func = toolbox.compile(expr=individual)

        #print(type(func))
        #print(individual)
        pred = func(x, y)
        label = objective(x, y)
        loss = fitness(pred, label)
        return loss,

    toolbox.register("evaluate", weight, data=data)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr,pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox