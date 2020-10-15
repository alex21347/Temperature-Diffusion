#Simulating and Analysing the Method of Relaxation for 2D Dirichlet Problem

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import cm
import time
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

#setting up for-loop to find average time to reach accurate solution
dum = 10
tic = time.time()
for i in tqdm(range(dum)):
    
    
    a = 17                              #length of grid 
    b = 17                              #width of grid
    its = 200                           #number of iterations
    p_x = np.zeros((b,a,its))
    x_coords = np.linspace(0,a-1,a)
    y_coords = np.linspace(0,b-1,b)
    graph = np.zeros((a*b,2))
    
    #generating 2D graph
    for j in range(0,a):
        for i in range(0,b):
            graph[a*i+j,0] = x_coords[j]
            graph[a*i+j,1] = y_coords[i]
     
    #Building the set of edges via finding closest neighbours of each vertex    
    neighbours = []
    g = graph
    for k in range(a*b):
        neighbours.append(np.array(np.where(abs(g[k,1]-g[:,1])+abs(g[k,0]-g[:,0])<1.1))[0])
        neighbours[k] = np.delete(neighbours[k], np.where(neighbours[k] == k), axis=0)
        
    vals = np.zeros((a*b,1))
    scenario = 16        #for scenario 1 set equal to 2, for scenario 2 set equal to 16
    vals[:scenario] = 1             
    
    #finding interior of graph i.e. where the walker may walk
    interior = []
    for i in range(a*b):
        if len(neighbours[i]) == 4:
            interior.append(i)
    
    #approximating p(x,y) over many iterations
    for i in range(its):
        for k in range(len(vals)):
            if k in interior:
                vals[k] = np.mean(vals[neighbours[k]])
                p_x[int((k-np.mod(k,a))/(a)),np.mod(k,a),i] = vals[k]
        if i > 0:  
           errorest = (np.abs(p_x[:,:,i]-p_x[:,:,i-1]).sum())/15**2
           p_x1 = p_x[1:-1,1:-1]
    
    #neatening solution            
    p_x = p_x[1:-1,1:-1]

toc = time.time()
print(f'Time Elapsed : {(toc-tic)/dum}')

#plotting solution of 2D Dirichlet Problem via Method of Relaxation

fig = plt.figure(figsize = (8,6))
ax = fig.gca(projection='3d')

Y = np.arange(0, b-2, 1)
Y = -1* Y
X = np.arange(0, a-2, 1)
X,Y = np.meshgrid(X, Y)
Z = np.transpose(p_x[:,:,99])

surf = ax.plot_surface(X,Y,Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()