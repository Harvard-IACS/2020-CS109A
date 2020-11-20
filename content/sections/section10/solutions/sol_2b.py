kernel_weight = 0.03
bias_weight = 0.03
model_iris_l1 = models.Sequential([
  layers.Input(shape = (4,)),
  layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1(kernel_weight), bias_regularizer=regularizers.l2(bias_weight)),
  layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1(kernel_weight), bias_regularizer=regularizers.l2(bias_weight)),
  layers.Dense(3, activation = 'softmax')
])
model_iris_l1.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizers.Adam(0.005),
    metrics=['accuracy'],
)
iris_trained_l1 = model_iris_l1.fit(
    x = X_train_iris.to_numpy(), y = y_train_iris.to_numpy(), verbose=0,
    epochs=1000, validation_data= (X_test_iris.to_numpy(), y_test_iris.to_numpy()),
)

plot_accuracy_loss_rolling(iris_trained_l1)

