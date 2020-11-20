epochs_max1 = 2000
n_neurons1  = 50
kernel_weight = 0.0008
bias_weight   = 0.0008

model_br1_l2 = models.Sequential(name='br1_overfit')

# first hidden layer
model_br1_l2.add(layers.Dense(n_neurons1, activation='tanh',
                            kernel_initializer='random_normal', bias_initializer='random_normal',
                            input_shape=(1,),
                            kernel_regularizer=regularizers.l2(kernel_weight),
                            bias_regularizer=regularizers.l2(bias_weight)))
# second hidden layer
model_br1_l2.add(layers.Dense(n_neurons1, activation='tanh',
                           kernel_regularizer=regularizers.l2(kernel_weight),
                           bias_regularizer=regularizers.l2(bias_weight)))
# third hidden layer
model_br1_l2.add(layers.Dense(n_neurons1, activation='tanh',
                           kernel_regularizer=regularizers.l2(kernel_weight),
                           bias_regularizer=regularizers.l2(bias_weight)))
# output layer, one neuron
model_br1_l2.add(layers.Dense(1,  activation='linear'))

adam = optimizers.Adam(lr=0.01) 

model_br1_l2.compile(loss='MSE',optimizer=adam)


history_br1_l2 = model_br1_l2.fit(x_train1, y_train1,
            validation_data=(x_test1,y_test1), epochs=epochs_max1, batch_size= 32, verbose=0)

plot_loss(history_br1_l2)
plot_sets(r1_train, r1_test, NN_model = model_br1_l2, title = "BreakRoom1")
