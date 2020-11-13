######################
##  Using ReLU
######################
# set the network parameters
w1 = 2
b1 = 0.0
w2  = 1
b2  = 0.5

# affine operation
l1 = w1*x_train + b1

# activation (Choose between ReLu or Sigmoid)
h = g(l1) # for relu

# output linear layer
y_model_train = w2*h + b2

# Make a plot the plot_toyModels function
plt.figure(figsize=[12,4])
plt.subplot(1,2,1)
plot_toyModels(x_train, y_train, y_model_train)
plt.title('ReLU activation')
# Evaluate the prediction
mse_toy = mean_squared_error(y_train, y_model_train)
print('ReLU: The MSE for the training set is ', np.round(mse_toy,5))



######################
##  Using sigmoid
######################
w1 = 6
b1 = -3.
w2  = 2
b2  = 0.5

# affine operation
l1 = w1*x_train + b1

# activation (Choose between ReLu or Sigmoid)
h = sig(l1) # for sigmoid

# output linear layer
y_model_train = w2*h + b2

# Make a plot the plot_toyModels function
plt.subplot(1,2,2)
plot_toyModels(x_train, y_train, y_model_train)
plt.title('Sigmoid activation')
# Evaluate the prediction
mse_toy = mean_squared_error(y_train, y_model_train)
print('Sigmoid: The MSE for the training set is ', np.round(mse_toy,5))
