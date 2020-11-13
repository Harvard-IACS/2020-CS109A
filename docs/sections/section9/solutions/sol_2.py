
#############################
# Design the neural network
#############################
model = models.Sequential(name='MyNet')

# hidden layer with 20 neurons (or nodes)
model.add(layers.Dense(20, activation='tanh', input_shape=(1,)))
# Add another hidden layer of 20 neurons
model.add(layers.Dense(20, activation='tanh'))
# output layer, one neuron 
model.add(layers.Dense(1,  activation='linear'))
# model.summary()

##############################################
## SET OPTIMIZER AND LOSS. COMPILE AND FIT
##############################################
optimizer=optimizers.Adam(0.005)
model.compile(loss='MSE',optimizer=optimizer) 
history_toy2 = model.fit(x_train2, y_train2, epochs=1000, batch_size=64, verbose=0,
                     validation_data= (x_test2, y_test2))

# PLOT THE LOSS FUNCTIONS
plot_loss(history_toy2)

# PLOT DATA AND PREDICTIONS
plot_sets(toy_train2, toy_test2, NN_model = model)
