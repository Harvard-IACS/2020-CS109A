
## Multiple regression
model_boston_1 = LinearRegression().fit(boston_train[["LSTAT", "RM"]], boston_train.MEDV)
print('Coefficients: ', model_boston_1.coef_)
print("R^2 on testing set  of the multiple regression model: ",  r2_score(boston_test.MEDV, model_boston_1.predict(boston_test[["LSTAT","RM"]])) )
print(' ')

## Multiple regression including Interaction Term

# Data engineering to include interaction term as a new predictor (feature)
boston["LSTAT*RM"] = boston["LSTAT"]*boston["RM"]
boston_train, boston_test = train_test_split(boston, train_size=0.7, random_state=42)

model_boston_1_inter = LinearRegression().fit(boston_train[["LSTAT", "RM", "LSTAT*RM"]], boston_train.MEDV)
print('Coefficients: ', model_boston_1_inter.coef_)
print("R^2 on testing set of the model including interaction:", r2_score(boston_test.MEDV, model_boston_1_inter.predict(boston_test[["LSTAT", "RM", "LSTAT*RM"]])))

# When do we want an interaction term with two continous variables?
# Answer: If we ask, “What is the effect of LSTAT on MEDV”, and the answer is “it depends on what RM equals.”
# https://www3.nd.edu/~rwilliam/stats2/l55.pdf

#Store R^2 for later investigation
R2_1 =  r2_score(boston_test.MEDV, model_boston_1.predict(boston_test[["LSTAT","RM"]]))
R2_1_inter = r2_score(boston_test.MEDV, model_boston_1_inter.predict(boston_test[["LSTAT", "RM", "LSTAT*RM"]]))
