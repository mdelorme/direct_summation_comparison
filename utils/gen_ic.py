import sys
import numpy as np

N = 1000 if len(sys.argv) < 2 else int(sys.argv[-1])

pos = np.random.rand(N, 3)
vel = np.zeros((N, 3))
m   = np.random.rand((N)) + 1.0

data = np.stack((pos[:,0], pos[:,1], pos[:,2], vel[:,0], vel[:,1], vel[:,2], m)).T

np.savetxt('ic_{}.dat'.format(N), data)


