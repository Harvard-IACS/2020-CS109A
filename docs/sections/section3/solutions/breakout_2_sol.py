
# Data engineering to include polynomial terms as predictors (features)
boston["LSTAT**2"] = boston["LSTAT"]**2
boston["LSTAT**3"] = boston["LSTAT"]**3
boston["RM**2"] = boston["RM"]**2
boston["RM**3"] = boston["RM"]**3
boston_train, boston_test = train_test_split(boston, train_size=0.7, random_state=42)


# Regression with 2nd order terms
model_boston_2 = LinearRegression().fit(boston_train[["LSTAT", "RM",  "LSTAT**2",  "RM**2"]], boston_train.MEDV)
R2_2= r2_score(boston_test.MEDV, model_boston_2.predict(boston_test[["LSTAT", "RM", "LSTAT**2",  "RM**2"]]))
print('Coefficients: ', model_boston_2.coef_)
print("R^2 on testing set including 2nd order terms:", R2_2)


# Regression with 2nd order terms including interaction term
print(' ')
model_boston_2_inter = LinearRegression().fit(boston_train[["LSTAT", "RM",  "LSTAT**2",  "RM**2", "LSTAT*RM"]], boston_train.MEDV)
R2_2_inter= r2_score(boston_test.MEDV, model_boston_2_inter.predict(boston_test[["LSTAT", "RM", "LSTAT**2",  "RM**2","LSTAT*RM"]]))
print('Coefficients: ', model_boston_2_inter.coef_)
print("R^2 on testing set including 2nd order terms:", R2_2_inter)


# Regression with 3nd order terms including interaction term
print(' ')
model_boston_3_inter = LinearRegression().fit(
        boston_train[["LSTAT", "RM",  "LSTAT**2",  "RM**2","RM**3", "LSTAT**3", "LSTAT*RM"]], boston_train.MEDV)
R2_3_inter= r2_score(boston_test.MEDV, model_boston_3_inter.predict(
        boston_test[["LSTAT", "RM", "LSTAT**2",  "RM**2","RM**3", "LSTAT**3", "LSTAT*RM"]]))
print('Coefficients: ', model_boston_3_inter.coef_)
print("R^2 on testing set including 2nd order terms:", R2_3_inter)

