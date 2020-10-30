tree_depth_range = range(1, 20, 2)
rf_val_acc = get_tree_pd(x_train, 
                        y_train, 
                        RandomForestClassifier(), 
                        tree_depth_range)
rf_mean_acc  = rf_val_acc.groupby("depth").mean()
rf_mean_acc["depth"] = list(tree_depth_range)