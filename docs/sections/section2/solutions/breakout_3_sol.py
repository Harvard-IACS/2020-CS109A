# linear regression
# from statsmodels.api import OLS
# import statsmodels.api as sm

def linreg_model(train_data,test_data):
    # sort
    sorted_train = train_data.sort_values(['x'])
    sorted_test = test_data.sort_values(['x'])

    x_train, x_test, y_train, y_test = sorted_train['x'], sorted_test['x'], sorted_train['y'], sorted_test['y']
    # Add constant to x data
    x_train_ca = sm.add_constant(x_train)
    x_test_ca = sm.add_constant(x_test)

    # Create Linear Regression object
    linreg_model = sm.OLS(y_train, x_train_ca)

    # Fit
    results = linreg_model.fit()

    # predict
    train_preds = results.predict(x_train_ca)
    test_preds = results.predict(x_test_ca)
    
    # find r^2
    r2_train = metrics.r2_score(y_train, results.predict(x_train_ca))
    r2_test = metrics.r2_score(y_test, results.predict(x_test_ca))
    
    return train_preds, test_preds, r2_train, r2_test

def plot_predictions2(k,train_data,test_data, knn_train_preds, knn_test_preds, linreg_train_preds, linreg_test_preds):

    # SubPlots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,6))
    axes[0].plot(train_data['x'], train_data['y'], 'bo',alpha = 0.5, label = 'Data' )
    axes[0].plot(train_data['x'], knn_train_preds,'m-', linewidth = 2, markersize = 10,  label = 'KNN Preds')
    axes[0].plot(train_data['x'], linreg_train_preds,'k-', linewidth = 2, markersize = 10, label = 'Linreg Preds')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title("Train Data")
    axes[0].legend()

    axes[1].plot(test_data['x'], test_data['y'], 'r*', alpha = 0.5, label = 'Data' )
    axes[1].plot(test_data['x'], knn_test_preds, 'y-', linewidth = 2, markersize = 10, label = 'KNN Preds')
    axes[1].plot(test_data['x'], linreg_test_preds, 'g-', linewidth = 2, markersize = 10,label = 'Test Preds')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title("Test Data")
    axes[1].legend()

    fig.suptitle("KNN vs Linear Regression")
    plt.show()

# get predictions
linreg_train_preds,linreg_test_preds, linreg_r2_train, linreg_r2_test = linreg_model(sim_train_data,sim_test_data)

# plot linreg predictions side by side with knn predictions
k=10
plot_predictions2(k, sim_sorted_train, sim_sorted_test, knn_train_preds[1], knn_test_preds[1], linreg_train_preds, linreg_test_preds)

# print r2 scores for knn with k=10 and linreg
print("R^2 Score of kNN on training set with k={}:".format(k), knn_r2_train_scores[1])
print("R^2 Score of kNN on testing set: with k={}".format(k), knn_r2_test_scores[1])
print("R^2 Score of linear regression on training set", linreg_r2_train)
print("R^2 Score of linear regression on testing set", linreg_r2_test)