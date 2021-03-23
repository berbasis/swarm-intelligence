import numpy as np
import random
import matplotlib.pyplot as plt
from math import cos, sin, pi, exp, radians, sqrt

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
           
def check_bound(sol):
	sol[sol > 100] = 100
	sol[sol < -100] = -100
	return sol

def firefly_algorithm(no):
	alpha = 0.5
	beta0 = 1
	betamin = 0.2
	gamma = 1
	max_gen = 20
	num_population = 20
	population = np.random.uniform(-100, 100, (num_population, 2))
	brightness = np.zeros(num_population)
	for i in range(num_population):
		brightness[i] = compute_function(population[i], no)
	for gen in range(max_gen):
		for i in range(num_population):
			for j in [x for x in range(num_population) if x != i]:
				if(brightness[j] <= brightness[i]):
					r = sqrt((population[i][0] - population[j][0])**2 + (population[i][1] - population[j][1])**2)
					beta = (beta0 - betamin) * exp(-gamma * r**2) + betamin
					population[i] = population[i] * (1 - beta) + population[j] * beta + (alpha  * (np.random.uniform(0, 1, 2) - 0.5) * 200)
					population[i] = check_bound(population[i])
					brightness[i] = compute_function(population[i], no)
		alpha = (1-(1 - (10 ** (-4) / 0.9) ** (1 / max_gen))) * alpha
	best_sol = np.where(brightness == np.max(brightness))
	best_sol = best_sol[0][0]
	return (population[best_sol], compute_function(population[best_sol], no))

if __name__ == '__main__':
	print("One time run:")
	for no in range(1, 3):	
		sol = firefly_algorithm(no)
		print(no, ". ", "x1 = ", sol[0][0], "\n   x2 = ", sol[0][1], "\n   f(x1, x2) = ", sol[1], "\n", sep="")
	print("30 times run average:")
	for no in range(1, 3):
		sol_total = np.zeros(3)
		for i in range(30):
			sol = firefly_algorithm(no)
			sol_total[0] += sol[0][0]
			sol_total[1] += sol[0][1]
			sol_total[2] += sol[1]
		print(no, ". ", "x1 = ", sol_total[0]/30, "\n   x2 = ", sol_total[1]/30, "\n   f(x1, x2) = ", sol_total[2]/30, "\n", sep="")
