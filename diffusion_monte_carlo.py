#Simulating and Analysing the Monte-Carlo Method for 2D Dirichlet Problem

import numpy as np
import time
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

a = 17              #length of grid
b = 17              #width of grid
N = 10000           #maximum number of steps walks may take before deletion
its  = 100          #number of random walk per starting position of grid

x_0s = np.zeros(((a-2)*(b-2),2))
p_x = np.zeros((a-2,b-2,its))
successes = np.zeros((a-2,b-2))

#building co-ordinate system
for i in range(a-2):
    for j in range(b-2):
        x_0s[(b-2)*i+j,0] = i
        x_0s[(b-2)*i+j,1] = j
        

def take_a_step(position):
    
    #walker takes a step
    
    steps = [[-1,0],[1,0],[0,1],[0,-1]]
    new_step = np.array(random.choice(steps))
    position = position + new_step
    
    return position            


#setting up for-loop to find average time to reach accurate solution
dum = 10
tic = time.time()
for p in tqdm(range(dum)):
    for j in tqdm(range(its)):
        for k in range((a-2)*(b-2)):
            success = 0
            pos = np.array([x_0s[k,0],x_0s[k,1]])
            
            #scenario 1 below - uncomment to use!
    # =============================================================================
    #         for i in range(0,N):
    #             pos = take_a_step(pos)
    #             if pos[0] == -1 or pos[0] == a-2 or pos[1] == b-2 :
    #                 break
    #             if pos[1] == -1 and pos[0]<a-3:
    #                 break
    #             if pos[1] == -1 and pos[0] == a-3:
    #                 successes[int((k-np.mod(k,b-2))/(b-2)),np.mod(k,b-2)] =
    #successes[int((k-np.mod(k,b-2))/(b-2)),np.mod(k,b-2)] + 1
    #                 break
    # =============================================================================
            
            #scenario 2 below - uncomment to use!
    # =============================================================================
    #         for i in range(0,N):
    #             pos = take_a_step(pos)
    #             if pos[0] == -1 or pos[0] == a-2 or pos[1] == b-2 :
    #                 break
    #             if pos[1] == -1:
    #                 successes[int((k-np.mod(k,b-2))/(b-2)),np.mod(k,b-2)] =
    #successes[int((k-np.mod(k,b-2))/(b-2)),np.mod(k,b-2)] + 1
    #                 break
    #             
    #     p_x[:,:,j] = successes/(j+1)
    # =============================================================================

toc = time.time()
print(f'Time Elapsed : {(toc-tic)/dum}')

#Plotting solution
    
fig = plt.figure(figsize = (8,6))
ax = fig.gca(projection='3d')

X = np.arange(0, b-2, 1)
Y = np.arange(0, a-2, 1)
X, Y = np.meshgrid(X, Y)
Z = p_x[:,:,9000]


surf = ax.plot_surface(X,Y,Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()