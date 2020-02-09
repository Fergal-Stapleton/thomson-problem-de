from __future__ import division
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from lib.diff_evolution import diff_evolv_3d
from lib.draw_plot import draw_sphere
from lib.potential import columb_pot

def main():
	# Declare immutable variables
	n = 12 # number of electrons
	gen_max = 200 # Max number of generations
	pop_num = 500 # number of spheres generated
	coordNum = 2 # Number of coordinates

	best_pot_range = [] # Track best and worst potential for graphs
	worst_pot_range = []

	inputFile = "data/thomsonDataSet.csv"

	# Read in example potentials to compare our results against
	df = pd.read_csv(os.path.expanduser(inputFile),  sep=',')
	df = df[df["N"].isin([n])]
	referencePot = df["Pot"]
	referenceN = df["N"]

	# Create diff evolution class
	de = diff_evolv_3d(gen_max, pop_num, coordNum, n)
	de.feval_init_population(pop_num, n)
	de.best_mem_init_eval(pop_num)
	de.best_mem_current = de.pop[de.best_index] # best member of current iteration

	# Initialize real time plot of best vs worst
	fig1 = plt.figure()
	plt.axis([0, gen_max, float(referencePot), float(de.best_val)*1.5])

	for gen_num in range(gen_max):
		de.gen_count = de.gen_count + 1

		print("Generation Number: "+str(de.gen_count))
		# create a reference population filled with the best member of the previous generation
		for k in range(pop_num):
			de.best_mem[k] = de.best_mem_current

		# f_weight decreases by gen number, uncomment below to keep constant weight
		if de.gen_count > 10:
			de.f_weight_calulation(gen_num, gen_max)

		if de.switch_var == 0:
			de.mutation(pop_num, n, coordNum)
		else:
			de.mutation_jitter(pop_num, n, coordNum)
		de.new_pop = de.trial_pop

		de.pop_reintroduction(pop_num, n)
		de.pop_sort(pop_num, n)

		de.selection(pop_num, n)
		de.worst_mem_per_gen(pop_num, n)

		best_current_potential = columb_pot(de.pop, de.best_index, n)
		worst_current_potential = columb_pot(de.pop, de.worst_index, n)

		print("Best potential after DE: ")
		print(best_current_potential)
		print("Best index of current iteration: ")
		print(de.best_index)
		print("Worst potential after DE: ")
		print(worst_current_potential)
		print("Best index of current iteration: ")
		print(de.worst_index)
		print("")


		de.best_per_gen[de.gen_count-1] = best_current_potential
		de.worst_per_gen[de.gen_count-1] = worst_current_potential

		best_pot_range.append(best_current_potential)
		worst_pot_range.append(worst_current_potential)

		gen_range = pd.Series(np.arange(de.gen_count))

		plt.plot(gen_range, pd.Series(best_pot_range), "b", gen_range, pd.Series(worst_pot_range), "r")
		plt.pause(0.05)


	#Call draw sphere function
	print("")
	print("Best Potential Found: ")
	print(columb_pot(de.pop, de.best_index,n))
	draw_sphere(de.best_mem_current, n)

main()
