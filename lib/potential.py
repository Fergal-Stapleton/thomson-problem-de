import numpy as np

# Coulomb potential
def coulomb_pot(pop, index, n):
	potential = 0.0
	for i in range(n-1):
		for j in range(i+1,n):
			potential = potential + (1/(np.sqrt(2-2*(np.cos(pop[index][i][0])*
										np.cos(pop[index][j][0]) +
										np.sin(pop[index][i][0])*
										np.sin(pop[index][j][0])*
										np.cos(pop[index][i][1]-pop[index][j][1])))))
	return potential
