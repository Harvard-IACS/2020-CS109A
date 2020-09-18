# read in data
data = pd.read_csv("../data/sim_data.csv")

# drop Unnamed column
data.drop(columns=["Unnamed: 0"], inplace=True)

# split into training and testing with 80/20 split, random_state=42
train_data1, test_data1 = train_test_split(data, test_size=0.20, random_state=42)

# plot results 80/20 split
plt.figure(figsize=[8,5])
plt.scatter(train_data1.x, train_data1.y, c="blue", marker='o', label="train data")
plt.scatter(test_data1.x, test_data1.y,c="red", marker='*', label="test data")
plt.xlabel('x')
plt.ylabel('y')
plt.title("80/20 Split")
plt.legend()
plt.show()

