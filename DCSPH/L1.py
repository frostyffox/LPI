https://colab.research.google.com/drive/1LhUOFerpNJPddzujAZYajYQVk6sSXOnw

import matplotlib.pyplot as plt
import numpy as np

# Plot data
x = np.linspace(-10, 15, 100)
y = 5* x + 20

# plot creation
fig, ax = plt.subplots(figsize=(10,6))

# main line (fn)
ax.plot(x,y, color = 'blue', linewidth=2, label = "Pollution over time")

# derivative
fprime = np.full_like(x,5)
ax.plot(x, fprime, 'r--', label="Derivatoive")