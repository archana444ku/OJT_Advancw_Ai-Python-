import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


#Q. 1. Load the dataset from a CSV file named flower.csv into a Pandas DataFrame. Display the first few rows of the dataset.
# Load the dataset
df = pd.read_csv('flower.csv')

# Display the first few rows
print(df.head())

#Q.2. Generate summary statistics for this dataset. What are the mean and standard deviation of the Sepal Length?
# Summary statistics
print(df.describe())

#Q. 3. Check for any missing values in the dataset. How would you handle them if there were any?
# Check for missing values
print(df.isnull().sum())

#Q. 4. Convert the species labels to numerical values using a mapping dictionary. For example, map 'FlowerA' to 0, 'FlowerB' to 1, and 'FlowerC' to 2.
# Mapping dictionary
species_map = {'FlowerA': 0, 'FlowerB': 1, 'FlowerC': 2}

# Apply mapping to the 'Species' column
df['Species'] = df['Species'].map(species_map)


#Q. 5. Split the dataset into training and testing sets with 70% training data and 30% testing data. Ensure that the split is stratified based on the species.
# Features (X) and target (y)
X = df.drop(columns=['Species'])
y = df['Species']

# Stratified split based on 'Species'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

#Q. 6. Train a decision tree classifier on the training data. What parameters would you use for the decision tree?
# Initialize the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)



# Fit the classifier on the training data
clf.fit(X_train, y_train)

#Q.7. Visualize the trained decision tree.
# Define class_names based on species_map
class_names = list(species_map.keys())

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=class_names)
plt.show()

# Now plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=class_names)
plt.show()



#Q. 8. Predict the species for the testing data and compute the accuracy.
# Predictions on the testing data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


#Q. 9. Generate a classification report and a confusion matrix for the predictions.
# Classification report
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=species_map.keys()))

# Confusion matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
