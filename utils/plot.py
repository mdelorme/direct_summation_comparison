import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

fname = sys.argv[-1]

dat = np.loadtxt(fname)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dat[:,0], dat[:,1], dat[:,2], s=1)

ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_zlim(0.0, 1.0)

plt.show()
