# from sklearn.neighbors import KNeighborsRegressor

# split into 70/30, random_state=42
sim_train_data, sim_test_data = train_test_split(data, test_size=0.30, random_state=42)

def knn_model(k, train_data, test_data):
    # create the classifier object
    neighbors = KNeighborsRegressor(n_neighbors=k)

    # fit the model using x_train as training data and y_train as target values
    neighbors.fit(train_data[['x']], train_data['y'])

    sorted_train = train_data.sort_values(['x'])
    sorted_test = test_data.sort_values(['x'])

    # Retreieve our predictions:
    train_preds = neighbors.predict(sorted_train[['x']])
    test_preds = neighbors.predict(sorted_test[['x']])
    
    # find r^2
    r2_train = neighbors.score(train_data[['x']], train_data['y'])
    r2_test = neighbors.score(test_data[['x']], test_data['y'])
    
    print("R^2 Score of kNN on training set with k={}:".format(k), r2_train)
    print("R^2 Score of kNN on testing set: with k={}".format(k), r2_test)
    return sorted_train, sorted_test, train_preds, test_preds, r2_train, r2_test

def plot_predictions_same_plot(k, train_data,test_data, train_preds, test_preds):
    # plot all results on same plot
    plt.figure(figsize=[8,6])
    plt.plot(train_data['x'], train_data['y'], 'bo', alpha = 0.5, label = 'Train Set' )
    plt.plot(train_data['x'], train_preds, 'k-', linewidth = 2, markersize = 10, label = 'Train Preds')
    plt.plot(test_data['x'], test_data['y'], 'r*', alpha = 0.5, label = 'Test Set' )
    plt.plot(test_data['x'], test_preds, 'g-', linewidth = 2, markersize = 10, label = 'Test Preds')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("x vs y kNN Regression (k={})".format(k))
    plt.legend()
    plt.show()

knn_train_preds = []
knn_test_preds = []
knn_r2_train_scores = []
knn_r2_test_scores = []

for k in [1,10,70]:
    sim_sorted_train, sim_sorted_test, sim_train_preds, sim_test_preds, knn_r2_train, knn_r2_test = knn_model(k, sim_train_data,sim_test_data)
    plot_predictions_same_plot(k,sim_sorted_train, sim_sorted_test, sim_train_preds, sim_test_preds)
    knn_train_preds.append(sim_train_preds)
    knn_test_preds.append(sim_test_preds)
    knn_r2_train_scores.append(knn_r2_train)
    knn_r2_test_scores.append(knn_r2_test)
