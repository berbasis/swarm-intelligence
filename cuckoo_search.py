import numpy as np
from math import cos, sin, pi, exp, radians, sqrt, gamma

def compute_function(sol, no):
	if(no == 1):
		h1 = 0 
		h2 = 0
		for i in range(1,6):
			h1 += (i * cos(radians((i+1) * sol[0] + 1)))
			h2 += (i * cos(radians((i+1) * sol[1] + 1)))
		return -h1 * h2
	elif(no == 2):
		return -cos(radians(sol[0])) * cos(radians(sol[1])) * exp(-((sol[0] - pi)**2) - ((sol[1] - pi)**2))
	return 0

def levy_flight(beta):
	sigma = np.power((gamma(1 + beta) * np.sin((np.pi * beta) / 2)) / gamma((1 + beta) / 2) * np.power(2, (beta - 1) / 2), 1 / beta)
	u = np.random.normal(0, sigma, size=2)
	v = np.random.normal(0, 1, size=2)
	step = u / np.power(np.fabs(v), 1 / beta)
	return step

def check_bound(sol):
	sol[sol > 100] = 100
	sol[sol < -100] = -100
	return sol

def cuckoo_search(no):
	alpha = 0.1
	beta = 1.0
	num_population = 50
	max_gen = 50
	nests = np.random.uniform(-100, 100, (num_population, 2))
	fitness = np.zeros(num_population)
	for i in range(num_population):
		fitness[i] = compute_function(nests[i], no)
	for gen in range(max_gen):
		for i in range(num_population):
			cuckoo = nests[i] + (alpha * levy_flight(beta))
			cuckoo = check_bound(cuckoo)
			cuckoo_fitness = compute_function(cuckoo, no)
			rand = np.random.randint(0, num_population)
			if(cuckoo_fitness < compute_function(nests[rand], no)):
				nests[rand] = cuckoo
				fitness[rand] = cuckoo_fitness
		fitness = fitness[fitness.argsort()]
		nests = nests[fitness.argsort()]
		for i in range(num_population//2, num_population):
			nests[i] = np.random.uniform(-100, 100, 2)
			fitness[i] = compute_function(nests[i], no)
	best_sol = np.where(fitness == np.min(fitness))
	best_sol = nests[best_sol[0][0]]
	return best_sol, compute_function(best_sol, no)

if __name__ == '__main__':
	print("One time run:")
	for no in range(1, 3):	
		sol = cuckoo_search(no)
		print(no, ". ", "x1 = ", sol[0][0], "\n   x2 = ", sol[0][1], "\n   f(x1, x2) = ", sol[1], "\n", sep="")
	print("30 times run average:")
	for no in range(1, 3):
		sol_total = np.zeros(3)
		for i in range(30):
			sol = cuckoo_search(no)
			sol_total[0] += sol[0][0]
			sol_total[1] += sol[0][1]
			sol_total[2] += sol[1]
		print(no, ". ", "x1 = ", sol_total[0]/30, "\n   x2 = ", sol_total[1]/30, "\n   f(x1, x2) = ", sol_total[2]/30, "\n", sep="")