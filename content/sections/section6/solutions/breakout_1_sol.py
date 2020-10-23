pca_iris = PCA().fit(iris_scaled)
iris_scaled_pca = pca_iris.transform(iris_scaled)
print('Original dimensions:', iris_scaled.shape)
print('PCA dimensions:     ', iris_scaled_pca.shape)

## PLOT THE EXPLAINED VARIANCE RATIO
fig, ax = plt.subplots(ncols=2, figsize=(20,6))
ax1, ax2 = ax.ravel()

ratio = pca_iris.explained_variance_ratio_
ax1.bar(range(len(ratio)), ratio, color='blue', alpha=0.8)
ax1.set_title('Explained Variance Ratio PCA', fontsize=20)
ax1.set_xticks(range(len(ratio)))
ax1.set_xticklabels(['PC {}'.format(i+1) for i in range(len(ratio))])
ax1.set_ylabel('Explained Variance Ratio')

# accumulated explained variance ratio
ratio = pca_iris.explained_variance_ratio_
ax2.plot(np.cumsum(ratio), 'o-')
ax2.set_title('Cumulative Sum of Explained Variance Ratio PCA', fontsize=20)
ax2.set_ylim(0,1.1)
ax2.set_xticks(range(len(ratio)))
ax2.set_xticklabels(['PC {}'.format(i+1) for i in range(len(ratio))])
ax2.set_ylabel('Cumulative Sum of Explained Variance Ratio');


#################
#Visualize in 2D
#################

flower_species = dataset.target_names

fig, ax = plt.subplots(figsize=(16,8))
for i in range(dataset.target.shape[0]):
    if dataset.target[i]==0:
        c='b'
    elif dataset.target[i]==1:
        c='r'
    else:
        c='g'
    ax.plot(iris_scaled_pca[i,0], iris_scaled_pca[i,1], 'o', 
            markersize=8, color=c, label = flower_species[dataset.target[i]])
    
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')

vecs_iris = pca_iris.components_[0:4].T

# plotting arrowheads
for i, vec in enumerate(vecs_iris):
    ax.arrow(0,0,vec[0],vec[1], color='black', head_width=0.1)
    s = 1.3
    ax.annotate(dataset.feature_names[i], (s*vec[0], s*vec[1]), color='black')

#remove duplicate labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()