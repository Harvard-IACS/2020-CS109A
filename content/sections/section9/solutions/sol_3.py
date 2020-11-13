##################
# MAKE THE MODEL
##################
model_iris = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape = (4,)),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(3, activation = 'softmax')
])
model_iris.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.005),
    metrics=['accuracy'],
)

##################
# TRAIN THE MODEL
##################
iris_trained = model_iris.fit(
    x = X_train.to_numpy(), y = y_train.to_numpy(), verbose=0,
    epochs=100, validation_data= (X_test.to_numpy(), y_test.to_numpy()),
)
plot_accuracy_loss(iris_trained)
