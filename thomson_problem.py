from __future__ import division
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as m3D
import random
import time


def columb_pot(pop, index, n):
	potential = 0.0
	for i in range(n-1):
		for j in range(i+1,n):
			potential = potential + (1/(np.sqrt(2-2*(np.cos(pop[index][i][0])*
										np.cos(pop[index][j][0]) +
										np.sin(pop[index][i][0])*
										np.sin(pop[index][j][0])*
										np.cos(pop[index][i][1]-pop[index][j][1])))))
	return potential

def drawSphere(best_mem_current, n):
	u = np.linspace(0, 2 * np.pi, 100)
	v = np.linspace(0, np.pi, 100)

	x = np.outer(np.cos(u), np.sin(v))
	y = np.outer(np.sin(u), np.sin(v))
	z = np.outer(np.ones(np.size(u)), np.cos(v))

	fig2 = plt.figure()
	ax2 = fig2.gca(projection='3d')

	ax2.plot_surface(x, y, z, rstride=3, cstride=3, color='b', alpha=0.3, shade=10)

	#Place electrons on sphere
	for k in range(n):
		X = np.cos(best_mem_current[k][1])*np.sin(best_mem_current[k][0])
		Y = np.sin(best_mem_current[k][1])*np.sin(best_mem_current[k][0])
		Z = np.cos(best_mem_current[k][0])
		ax2.scatter(X, Y, Z, color='r')

	#import matplotlib.animation as animation

	#def rotate(angle):
	#	ax2.view_init(azim=angle)

	#rotating_sphere = animation.FuncAnimation(fig2, rotate, frames=np.arange(0, 360, 5), interval=100)
	#rotating_sphere.save('sphere.gif', dpi=160, writer='imagemagick')

class diff_evolv:
	def __init__(self, gen_max, pop_num, coordNum, n):
		# Declare mutable variables
		self.gen_count = 0; # Initialize Generation count
		self.f_CR = 0.5 # Crossover probability
		self.f_weight = 0.3 # Weighted function
		self.best_index = 1 # start with first population member as candidate for best, will evaluate against other population members
		self.worst_index = 1 # start with first population member as candidate for worst, will evaluate against other population members
		self.switch_var = 1

		self.std_dev_second = [[0 for x in range(1)] for x in range(gen_max)]

		self.best_per_gen = [[0 for x in range(1)] for x in range(gen_max)]
		self.worst_per_gen = [[0 for x in range(1)] for x in range(gen_max)]
		self.feval = [[0 for x in range(1)] for x in range(pop_num)]
		self.val = [[0 for x in range(1)] for x in range(pop_num)]
		self.tempval = [[0 for x in range(1)] for x in range(pop_num)]
		self.best_val = [[0 for x in range(1)] for x in range(pop_num)]
		self.worst_val = [[0 for x in range(1)] for x in range(pop_num)]
		self.best_mem_current = [[0 for x in range(1)] for x in range(pop_num)]


		# Create 3D population matrix in form [[[ Phi, Theta] electron number ] population number ]
		self.pop = [[[0 for x in range(coordNum)]  for x in range(n)] for x in range(pop_num)]
		self.best_mem = [[[0 for x in range(coordNum)]  for x in range(n)] for x in range(pop_num)]
		self.new_pop = [[[0 for x in range(coordNum)]  for x in range(n)] for x in range(pop_num)]
		self.trial_pop = [[[0 for x in range(coordNum)]  for x in range(n)] for x in range(pop_num)]


	def feval_init_population(self, pop_num, n):
		for index in range(pop_num):
			# First point is fixed for all spheres.
			self.pop[index][0][0] = np.pi/2
			self.pop[index][0][1] = 0
			# The rest of the points are randomly generated.
			for k in range(n-1):
				u = random.uniform(0, 1)
				self.pop[index][k+1][0] = np.arccos(2*u-1) # Phi value
				v = random.uniform(0, 1)
				self.pop[index][k+1][1] = 2*np.pi*v # Theta value
			self.feval[index] = columb_pot(self.pop, index, n)

	def best_mem_init_eval(self, pop_num):
		self.val[1] = self.feval[self.best_index] # function evaluation - Potential function
		self.best_val = self.val[1] # best objective function value so far
		self.worst_val = self.val[1] # worst objective function value so far
		for index in range(1, pop_num): # check the remaining members, 0 indexing so start at 1 as not to evaluate against initial.best_val
			self.val[index] = self.feval[index]
			if (self.val[index] < self.best_val): # find when.val <.best_val
				self.best_index = index # save its location
				self.best_val = self.val[index] # switch best_val
			if (self.val[index] > self.worst_val):
				self.worst_val = self.val[index]
			self.best_mem_current = self.pop[self.best_index] # best member of current iteration

	def worst_mem_per_gen(self, pop_num, n):
		for index in range(pop_num):
			self.feval[index] = columb_pot(self.pop, index, n)
			self.worst_index = 1
			self.val[1] = self.feval[self.worst_index]
			self.worst_val = self.val[1]
			for index in range(1, pop_num): # check the remaining members, 0 indexing so start at 1 as not to evaluate against initial.worst_val
				self.val[index] = self.feval[index]
				if (self.val[index] > self.worst_val): # find when.val >.worst_val
					self.worst_index = index # save its location
					self.worst_val = self.val[index] # switch worst_val

	def mutation(self, pop_num, n, coordNum):
		# Reinitialize trial pop as zeros
		self.trial_pop = [[[0 for x in range(coordNum)]  for x in range(n)] for x in range(pop_num)]
		for index in range(pop_num):
			A=0
			B=0
			C=0
			# randomly chose indexes for time being
			while( A == B ):
				A=int(np.ceil(random.uniform(0, 1)*pop_num)-1)
				B=int(np.ceil(random.uniform(0, 1)*pop_num)-1)
				C=int(np.ceil(random.uniform(0, 1)*pop_num)-1)

		for k in range(n):
			self.trial_pop[index][k][0] = self.best_mem[index][k][0]+(self.pop[A][k][0] - self.pop[B][k][0])*(self.f_weight)
			self.trial_pop[index][k][1] = self.best_mem[index][k][1]+(self.pop[A][k][1] - self.pop[B][k][1])*(self.f_weight)
			self.trial_pop[index][k][0], self.trial_pop[index][k][1] = self.boundary_constraint(index, k)

	def mutation_jitter(self, pop_num, n, coordNum):
		# We want random_pop to be randomized each time the strategy is called
		random_pop = [[[0 for x in range(coordNum)]  for x in range(n)] for x in range(pop_num)]
		self.trial_pop = [[[0 for x in range(coordNum)]  for x in range(n)] for x in range(pop_num)]

		for index in range(pop_num):
			random_pop[index][0][0]= np.pi/2; #phi [0 - pi]
			random_pop[index][0][1]= 0; #theta [0 - 2*pi]
			# The rest of the points are random_poply generated.
			for k in range(n-1):
				u = random.uniform(0, 1)
				random_pop[index][k+1][0] = np.arccos(2*u-1) # Phi value
				v = random.uniform(0, 1)
				random_pop[index][k+1][1] = 2*np.pi*v # Theta value

			A=0
			B=0
			# randomly chose indexes for time being
			while( A == B ):
				A=int(np.ceil(random.uniform(0, 1)*pop_num)-1)
				B=int(np.ceil(random.uniform(0, 1)*pop_num)-1)

			for k in range(n):
				self.trial_pop[index][k][0] = self.best_mem[index][k][0]+(self.pop[A][k][0] - self.pop[B][k][0])*((1-0.9999)*random_pop[index][k][0]+self.f_weight)
				self.trial_pop[index][k][1] = self.best_mem[index][k][1]+(self.pop[A][k][1] - self.pop[B][k][1])*((1-0.9999)*random_pop[index][k][1]+self.f_weight)
				self.trial_pop[index][k][0], self.trial_pop[index][k][1] = self.boundary_constraint(index, k)

	def boundary_constraint(self, index, k):
		# Insure outputs for selection are within bounds for phi and theta
		if self.trial_pop[index][k][0] < 0:
			self.trial_pop[index][k][0] += np.pi
		if self.trial_pop[index][k][0] > np.pi:
			self.trial_pop[index][k][0] -= np.pi
		if self.trial_pop[index][k][1] < 0:
			self.trial_pop[index][k][1] += 2*np.pi
		if self.trial_pop[index][k][1] > 2*np.pi:
			self.trial_pop[index][k][1] -= 2*np.pi
		return self.trial_pop[index][k][0], self.trial_pop[index][k][1]

	def selection(self, pop_num, n):
		for index in range(pop_num):
			self.feval[index] = columb_pot(self.new_pop, index, n)
			self.tempval = self.feval[index] # check cost of competitor
			if (self.tempval < self.val[index]):
				self.pop[index] = self.new_pop[index];  # replace old vector with new one (for new iteration)
				self.val[index] = self.tempval # save value in "cost array"
				if (self.tempval < self.best_val):
					self.best_index = index
					self.best_val = self.tempval; # new best value
					self.best_mem_current = self.new_pop[self.best_index] # new best parameter vector ever
					self.pop[self.best_index] = self.new_pop[self.best_index]

	def pop_sort(self, pop_num, n):
		for index in range(pop_num):
			## for k = 1:pop_num
			self.feval[index] = columb_pot(self.pop, index, n)

		# Sort best 10% of population members to retain into next generation
		pop_retain = int(np.ceil(0.1*pop_num))
		for index in range(pop_num+1):
			for index2 in range(pop_num-1):
				if self.feval[index2] > self.feval[index2+1]:
					self.pop[index2][:][:], self.pop[index2+1][:][:] = self.pop[index2+1][:][:], self.pop[index2][:][:]
					tempfeval = self.feval[index2]
					self.feval[index2] = self.feval[index2+1]
					self.feval[index2+1] = tempfeval


	def pop_reintroduction(self, pop_num, n):
		# Reintroduce 60% of the population using random coordinates
		pop_retintro_random = int(np.ceil(0.6*pop_num))
		for index in range(pop_retintro_random, pop_num):
			for k in range(n):
				u = random.uniform(0, 1)
				self.new_pop[index][k][0] = np.arccos(2*u-1) # Phi value
				v = random.uniform(0, 1)
				self.new_pop[index][k][1] = 2*np.pi*v # Theta value

	def f_weight_calulation(self, gen_num, gen_max):
		#std_dev_list = np.array([10])
		std_dev_list = self.best_per_gen[(gen_num-10):gen_num]
		self.std_dev_second[gen_num] = [np.std(std_dev_list)]
		stdDevTest = self.std_dev_second[(gen_num-10):gen_num]

		#print(std_dev_list)
		print("Standard Deviation of last 10 Generations: ")
		print(np.std(std_dev_list))
		print(np.std(stdDevTest))

		# test for convergence, if last 10 generations have not produced a significant deviation in potential adapt weight randomly and strategy to alternative
		if np.std(stdDevTest) < 0.00000001:
			self.f_weight = 0.1 * 0.9*(random.uniform(0, 1))
			if self.switch_var == 0:
				self.switch_var =1
			elif self.switch_var == 1:
				self.switch_var = 0
			print("Weight and Strategy Switch Occurred")


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
	de = diff_evolv(gen_max, pop_num, coordNum, n)
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
	drawSphere(de.best_mem_current, n)

main()
