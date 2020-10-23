# Performing PCA and asking for 2 principal components
pca_iris_2 = PCA(n_components=2).fit(iris_scaled)
iris_scaled_pca_2 = pca_iris_2.transform(iris_scaled)
print('Original dimensions:', iris_scaled.shape)
print('PCA dimensions:     ', iris_scaled_pca.shape)

# Split the dataset
X_train, X_test, y_train_iris, y_test_iris = train_test_split(iris_scaled_pca_2, dataset.target , test_size=0.4, random_state=42)
print('Shapes for X and y training sets:', X_train.shape, y_train_iris.shape)
print('Shapes for X and y testing sets:', X_test.shape, y_test.shape)

#Training a logistic regression model
model_logistic = LogisticRegression(C=100).fit(X_train, y_train_iris)

#Predict
y_pred_train = model_logistic.predict(X_train)
y_pred_test = model_logistic.predict(X_test)

#Perfromance Evaluation
train_score = accuracy_score(y_train_iris, y_pred_train)*100
test_score = accuracy_score(y_test_iris, y_pred_test)*100
print("Training Set Accuracy:",str(train_score)+'%')
print("Testing Set Accuracy:",str(test_score)+'%')

