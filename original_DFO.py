# Dispersive Flies Optimisation

# Reference to origianl paper:
# al-Rifaie, M. M. (2014). Dispersive flies optimisation. In 2014 Federated Conference on Computer Science and Information Systems (pp. 529-538). IEEE.

import numpy as np
import sys

# FITNESS FUNCTIONS
def sphere(x):
    return np.sum(np.array(x)**2)

def rosenbrock(x):
    x = np.asarray(x)
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1.0)**2)

def rastrigin(x):
    x = np.asarray(x)
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def ackley(x):
    x = np.asarray(x)
    n = len(x)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
            - np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
            + 20 + np.e)

FUNCTIONS = {
    'sphere': sphere,
    'rosenbrock': rosenbrock,
    'rastrigin': rastrigin,
    'ackley': ackley,
}

if len(sys.argv) != 5:
    print("Usage: python original_DFO.py [function_name] [population] [dimensions] [iterations]")
    print("  function_name: rastrigin, sphere, rosenbrock, ackley")
    sys.exit(1)

func_name = sys.argv[1]
if func_name not in FUNCTIONS:
    print(f"Unknown function '{func_name}'. Choose from: {', '.join(FUNCTIONS)}")
    sys.exit(1)

fitness_func = FUNCTIONS[func_name]
N = int(sys.argv[2])			# POPULATION SIZE
D = int(sys.argv[3])			# DIMENSIONALITY
maxIterations = int(sys.argv[4])	# ITERATIONS ALLOWED

# Standard benchmark bounds
BOUNDS = {
    'sphere':     (-100.0,   100.0),
    'rosenbrock': ( -30.0,    30.0),
    'rastrigin':  (  -5.12,    5.12),
    'ackley':     ( -32.768,  32.768),
}

delta = 0.001			# DISTURBANCE THRESHOLD
lower, upper = BOUNDS[func_name]
lowerB = [lower]*D		# LOWER BOUND (IN ALL DIMENSIONS)
upperB = [upper]*D		# UPPER BOUND (IN ALL DIMENSIONS)

# INITIALISATION PHASE
X = np.empty([N,D]) # EMPTY FLIES ARRAY OF SIZE: (N,D)
fitness = [None]*N  # EMPTY FITNESS ARRAY OF SIZE N

# INITIALISE FLIES WITHIN BOUNDS
for i in range(N):
	for d in range(D):
		X[i,d] = np.random.uniform(lowerB[d], upperB[d])

# MAIN DFO LOOP
for itr in range (maxIterations):
	for i in range(N): # EVALUATION
		fitness[i] = fitness_func(X[i,])
	s = np.argmin(fitness) # FIND BEST FLY

	if (itr%100 == 0): # PRINT BEST FLY EVERY 100 ITERATIONS
		print ("Iteration:", itr, "\tBest fly index:", s,
			   "\tFitness value:", fitness[s])

	# TAKE EACH FLY INDIVIDUALLY
	for i in range(N):
		if i == s: continue # ELITIST STRATEGY

		# FIND BEST NEIGHBOUR
		left = (i-1)%N
		right = (i+1)%N
		bNeighbour = right if fitness[right]<fitness[left] else left

		for d in range(D): # UPDATE EACH DIMENSION SEPARATELY
			if (np.random.rand() < delta):
				X[i,d] = np.random.uniform(lowerB[d], upperB[d])
				continue;

			u = np.random.rand()
			X[i,d] = X[bNeighbour,d] + u*(X[s,d] - X[i,d])

			# OUT OF BOUND CONTROL
			if X[i,d] < lowerB[d] or X[i,d] > upperB[d]:
				X[i,d] = np.random.uniform(lowerB[d], upperB[d])

for i in range(N): fitness[i] = fitness_func(X[i,]) # EVALUATION
s = np.argmin(fitness) # FIND BEST FLY

print("\nFinal best fitness:\t", fitness[s])
# print("\nBest fly position:\n",  X[s,])
