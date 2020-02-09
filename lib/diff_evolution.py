import numpy as np
import random
from lib.potential import columb_pot

class diff_evolv_3d:
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
		for index in range(1, pop_num): # check the remaining members, 0 indexing so start at 1 as not to evaluate against initial best_val
			self.val[index] = self.feval[index]
			if (self.val[index] < self.best_val): # find when val < best_val
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
			for index in range(1, pop_num): # check the remaining members, 0 indexing so start at 1 as not to evaluate against initial worst_val
				self.val[index] = self.feval[index]
				if (self.val[index] > self.worst_val): # find when val > worst_val
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

		# test for convergence, if last 10 generations have not produced a significant deviation
		# in potential adapt weight randomly and strategy to alternative
		if np.std(stdDevTest) < 0.00000001:
			self.f_weight = 0.1 * 0.9*(random.uniform(0, 1))
			if self.switch_var == 0:
				self.switch_var =1
			elif self.switch_var == 1:
				self.switch_var = 0
			print("Weight and Strategy Switch Occurred")
