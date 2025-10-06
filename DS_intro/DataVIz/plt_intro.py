# NOTEBOOK GOAL: SET UP A CANVAS FOR DATAVIZ

# Data
import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline 

x = np.arange(0,100) #arange: Return evenly spaced values within a given interval.
y = x*2
z = x**2

# exercise 1

# create a figure object called fig 
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(x,y, color= 'blue')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Linear Plot')
plt.show()

# create figure object and put two axes on it 
fig2 = plt.figure()
ax1 = fig.add_axes([0,0,1,1])
ax2 = fig.add_axes([0.2,0.5, 0.3 , 0.3])

# plot on both axes
ax1.plot(x,y, label = "y = 2x")
ax1.set_title("Main + Inset")
#ax1.set_xlim

ax2.plot(y,y, color="red", label = "y vs y")
#ax2.set_ylim


plt.show()

# create a plot adding two axes to a figure object at [0.0,1]
fig3 = plt.figure()
ax_main = fig3.add_axes([0,0,1,1])
ax_extra = fig3.add_axes([0.0, 0.1, 0.8, 0.8])

ax_main.plot(x,z, color = "green", label = "y = x^2")
ax_main.set_title("Quadratic")

ax_extra.plot(x,y, color = "orange", label = "y = 2x")
ax_main.set_title("Overlay")
plt.show()

# TBC...
