## FINISHEEED
# "Dummy regression task"
# our teacher doesnt have very high hopes for us :')

# 1: GENERATING DATA
# y(x) = 1.3x^3 - 3x^2 + 3.6x + 6.9
# goal: take random points in a range, uniformly
import numpy as np

np.random.seed(20) # random seed of 20 elements

def base_function(x):
    y = 1.3 * x ** 3 - 3 * x ** 2 + 3.6 * x + 6.9
    return y

low, high = -1, 3 #range extremes
n_points = 100

xs = np.random.uniform(low, high, n_points)[:,None] 
# random data: contains the right  points but also some of the ones we don't need

sample_ys = base_function(xs) # apply the target function to the data to get the real answer to the points

ys_noise = np.random.normal(loc =0.0, scale=1.0, size=len(xs))[:,None] #gaussian noise
noisy_sample_ys = sample_ys + ys_noise #add the noisy data to the "right" ys

# 2: PLOTTING DATA
import matplotlib.pyplot as plt

#plotting base function for range of interest
lsp = np.linspace(low,high)[:,None]
true_ys = base_function(lsp)
plt.plot(lsp, true_ys, linestyle='dashed', color='red')

# plot simulated data points
plt.scatter(xs, noisy_sample_ys, color = "green") #draw a line
plt.xlabel('x')
plt.ylabel('y')
#plt.show()



# 3: LINEAR REGRESSION (FITTING A SIMPLE MODEL)
from sklearn import linear_model

#xs = xs[:,None]
model = linear_model.LinearRegression() #define a model
model.fit(xs, noisy_sample_ys) # fit the model -> learning
#optimization process: finding the best line ..


predicted_lsp = model.predict(lsp) #see what did the model learn for each ppint
plt.scatter(xs, noisy_sample_ys)
plt.plot(lsp, predicted_lsp)
#plt.show()

#findings: the line is not the best model

# ASSESS HOW GOOD/BAD IS THE MODEL (Quantify)

#1 mean absolute difference between the samples
predictions = model.predict(xs)

# mean absolute error
# = average absolute difference btw pred.vals and actual target vals
# gives equal weight to all errors
# useful to: understand errors magnitude without considering over/under estim
mae = np.mean(np.abs(predictions - noisy_sample_ys))


# Normalised mean absolute error
n_mae = mae / np.mean(np.abs(noisy_sample_ys))

print("Mean absolute error: ", mae, "\nNormalised MAE: ", n_mae)

#  new set of labels with scale 3
ys_new_noise = np.random.normal(loc =0.0, scale=3.0, size=len(xs))[:,None] #gaussian noise
new_noisy_sample_ys = sample_ys + ys_new_noise
mae2 = np.mean(np.abs(predictions - new_noisy_sample_ys))

# Normalised mean absolute error
n_mae2 = mae2 / np.mean(np.abs(new_noisy_sample_ys))
print("Mean absolute error 2: ", mae2, "\nNormalised MAE 2: ", n_mae2)

from sklearn.neural_network import MLPRegressor

mlp_model = MLPRegressor(max_iter= 5000)
mlp_model.fit(xs, noisy_sample_ys)
predicted_lsp = mlp_model.predict(lsp)
predicted_train = mlp_model.predict(xs)
plt.scatter(xs, noisy_sample_ys)
plt.plot(lsp, predicted_lsp, color='orange', lw=2)
plt.show()

mae_ml = np.mean(np.abs(predicted_train - noisy_sample_ys.ravel())) #.ravel() flattens the labels to 1D
n_mae_ml = mae_ml / np.mean(np.abs(noisy_sample_ys))
print("WITH ML:\nMean absolute error: ", mae_ml, "\nNormalised MAE: ", n_mae_ml)
