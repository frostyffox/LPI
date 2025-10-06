# DATA
# titanic data set

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid") #alternatives: darkgrid

print(" ******** EXERCISE 1 ********")
print(" ******** Titanic Data Set ********")

titanic = sns.load_dataset('titanic')

print(titanic.head())

# _____ Joint Plot
# ex1: given a plot image, recreate it with code
# plot type: scatter plot combined to histogram
g = sns.jointplot(data=titanic, x="fare", y="age")
g.ax_joint.set_xlim(-100,600)
g.ax_joint.set_ylim(0,80)
# x axis: 'fare'; values between -100 and 600
# y axis: 'age'; between 0 and 80;

plt.show()

# ____Histogram
h = sns.displot(data=titanic, x="fare", col="sex", color='red', bins='auto')  
h.ax_joint.set_xlim(0,600)
h.ax_joint.set_ylim(0,500)


# ____Box Plot
b = sns.boxplot(data=titanic, x = "class", y="age", linewidth= 2, palette = "pastel")
plt.show()
