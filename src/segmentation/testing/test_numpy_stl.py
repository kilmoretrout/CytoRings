import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# attach to logger so trimesh messages will be printed to console
trimesh.util.attach_to_log()

mesh = trimesh.load('Prusa_Research_Vase_Dominik_Cisar.stl')
print(mesh.vertices)

print(mesh.vertices.shape, len(list(set(mesh.vertices[:,2]))))

#ax.scatter(mesh.vertices[:,0], mesh.vertices[:,1], mesh.vertices[:,2])

#plt.show()