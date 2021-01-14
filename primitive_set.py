from deap import gp

from utils import numpy_add, numpy_minus, numpy_mul

pset = gp.PrimitiveSet("MAIN", 2)
pset.addPrimitive(numpy_add, 2)
pset.addPrimitive(numpy_minus, 2)
pset.addPrimitive(numpy_mul, 2)
#pset.addEphemeralConstant("constant", lambda: np.random.randint(low=-2,high=2))
pset.renameArguments(ARG0="x")
pset.renameArguments(ARG1="y")