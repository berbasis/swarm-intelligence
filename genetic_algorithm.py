import numpy as np
import random
from math import cos, sin, pi, exp, radians

def init_binary_population(population_num, gene_num):
	return np.fix(2 * np.random.rand(population_num, gene_num)).astype(int)

def decode_chromosome(chromosome, ra, rb):
	max_sum = 0
	x = np.zeros((len(chromosome), 2))
	for kk in range(len(chromosome[0])//2):
		max_sum += 2**(kk)
	for i in range(len(chromosome)):
		for j in range(len(chromosome[i])//2):
			x[i,0] += chromosome[i][j] * (2**(len(chromosome[i])//2 - 1 - j))
			x[i,1] += chromosome[i][j + len(chromosome[i])//2] * (2**(len(chromosome[i])//2 - 1 - j))
		x[i,0] = binary_to_gray(int(x[i,0]))
		x[i,1] = binary_to_gray(int(x[i,1]))
		x[i,0] = (rb + ((ra - rb) * (x[i,0] / max_sum)))
		x[i,1] = (rb + ((ra - rb) * (x[i,1] / max_sum)))
	return x

def binary_to_gray(n):
	gray = 0
	while(n): 
		gray = gray ^ n; 
		n = n >> 1; 
	return gray

def compute_fitness(x1, x2, no):
	h = compute_function(x1, x2, no)
	return 2**-h

def compute_function(x1, x2, no):
	if(no == 1):
		h1 = 0 
		h2 = 0
		for i in range(1,6):
			h1 += (i * cos(radians((i+1)*x1 + 1)))
		for j in range(1,6):
			h2 += (j * cos(radians((j+1)*x2 + 1)))
		return -h1 * h2
	elif(no == 2):
		return -cos(radians(x1)) * cos(radians(x2)) * exp(-((x1 - pi)**2) - ((x2 - pi)**2))
	return 0

def tournament_selection(population, fitness, parents_num):
	parents = np.zeros((parents_num, len(population[0])))
	for i in range(parents_num):
		tournament = np.zeros((5, len(population[0])))
		tournament_fitness = np.zeros((5, len(population[0])))
		for j in range(5):
			idx = random.randint(0, len(population)-1)
			tournament[j] = population[idx]
			tournament_fitness[j] = (fitness[idx])
		best_fitness = np.where(tournament_fitness == np.max(tournament_fitness))
		best_fitness = best_fitness[0][0]
		parents[i] = tournament[best_fitness]
	return parents

def uniform_recombination(population, parents, offspring_num):
	offspring = np.zeros((offspring_num, len(population[0])))
	for i in range(0, offspring_num, 2):
		tp = np.fix(2 * np.random.rand(len(population[0])))
		for j in range(len(tp)):
			if (tp[j] == 0):
				offspring[i][j] = parents[i][j]
				offspring[i+1][j] = parents[i+1][j]
			elif (tp[j] == 1):
				offspring[i][j] = parents[i+1][j]
				offspring[i+1][j] = parents[i][j]
	return offspring

def flip_bit_mutation(population, mutation_probability):
	mutation = population
	for i in range(len(population)):
		for j in range(len(population[i])):
			if (random.random() < mutation_probability):
				if (population[i, j] == 0):
					population[i, j] = 1
				else:
					population[i, j] = 0
	return mutation

def genetic_algorithm(population_num, gene_num, mutation_probability, no):
	population = init_binary_population(population_num, gene_num)
	fitness = np.zeros(len(population))
	for i in range(10000//population_num):
		population_decoded = decode_chromosome(population,100,-100)
		for i in range(len(population_decoded)):
			fitness[i] = compute_fitness(population_decoded[i][0],population_decoded[i][1], no)
		parents = tournament_selection(population, fitness, population_num//2)
		offspring = uniform_recombination(population,parents, population_num//2)
		offspring = flip_bit_mutation(offspring, mutation_probability)
		population[0:len(parents)] = parents
		population[len(parents):] = offspring
	best_fitness = np.where(fitness == np.max(fitness))
	best_fitness = best_fitness[0][0]
	best_chromosome = population[best_fitness]
	best_solution = population_decoded[best_fitness]
	return best_chromosome, best_solution

if __name__ == '__main__':
	print("1 time run:\n")
	for no in range(1,3):
		best_chromosome, best_solution = genetic_algorithm(100, 30, 0.2, no)
		print(no, ". Best Chromosome: ", best_chromosome, sep="")
		print("   x1:", best_solution[0])
		print("   x2:", best_solution[1])
		print("   f(x1,x2):", compute_function(best_solution[0],best_solution[1],no), "\n")

	print("30 times average run:\n")
	for no in range(1,3):
		avg_solution = np.zeros(*best_solution.shape)
		for i in range(30):
			best_chromosome, best_solution = genetic_algorithm(100, 30, 0.2, no)
			avg_solution += best_solution
		avg_solution /= 30
		print(no, ". Best Chromosome: ", best_chromosome, sep="")
		print("   x1:", avg_solution[0])
		print("   x2:", avg_solution[1])
		print("   f(x1,x2):", compute_function(avg_solution[0],avg_solution[1],no), "\n")
		