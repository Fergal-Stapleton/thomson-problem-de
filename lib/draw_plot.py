import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as m3D

# Creates 3D sphere and assigns points to surface based on
# calculated coordinates.
def draw_sphere(best_mem_current, n):
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

	plt.show()

	# Below code can be uncommented if ImageMagick binaries are installed.
	# This will produce the gif file as seen in README.md file
	# -------------------------------------------------------------------

	#import matplotlib.animation as animation

	#def rotate(angle):
	#	ax2.view_init(azim=angle)

	#rotating_sphere = animation.FuncAnimation(fig2, rotate, frames=np.arange(0, 360, 5), interval=100)
	#rotating_sphere.save('images/sphere.gif', dpi=160, writer='imagemagick')
