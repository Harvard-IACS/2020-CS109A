#1
tree_depth_range = range(1, 15, 1)
cv_acc_pd = get_tree_pd(x_train, 
                        y_train, 
                        DecisionTreeClassifier(random_state = 42), 
                        tree_depth_range)
#2
plt.figure(figsize=(12, 3))
plt.title('Variation of Accuracy on Validation set with Depth - Simple Decision Tree')
sns.boxenplot(x = "depth", y = "cv_acc_score", data = cv_acc_pd);
plt.show()

#3
cv_acc_mean = cv_acc_pd.groupby("depth").mean()
cv_acc_mean["depth"] = list(tree_depth_range)

#4
plt.figure(figsize=(12, 3))
plt.title('Mean Validation set accuracy score — simple decision tree')
sns.lineplot(x = "depth", y = "cv_acc_score", data = cv_acc_mean);