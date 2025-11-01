import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error

data = pd.read_excel('customers.xlsx')

claims = data[data['Claim_Made'] == 'Y']
print("Claims dataframe:")
print(claims.head())


# Fit a decision tree to our data, comparing Age and Claim_Value in the dataframe that will be called claims
#  Form arrays x and y corresponding to these two columns of data by reshaping them.
x = claims['Age'].values.reshape(-1, 1)
y = claims['Claim_Value'].values

#  Split the data into training sets and test sets using the train_test_split function from sklearn.model_selection.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Use a DecisionTreeRegressor to fit the training data. Start with max_depth=3.
regressor = DecisionTreeRegressor(max_depth=3)
regressor.fit(x_train, y_train)

# Plot your decision tree
plt.figure(figsize=(12, 8))
plot_tree(regressor, filled=True, feature_names=['Age'])
plt.title("Decision Tree Regression")
plt.show()

# Calculate the average error of predictions compared to the actual test values.
y_pred = regressor.predict(x_test)
error = mean_absolute_error(y_test, y_pred)
print(f"Average error of predictions compared to actual test values: {error}")

# Plot the scatterplot of Claim_Value against Age.
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Claim_Value', data=claims)
plt.title("Scatterplot of Claim_Value against Age")
plt.xlabel("Age")
plt.ylabel("Claim Value")
plt.show()

# Add a column to the claims dataframe of the predicted values from your regressor for all customers.
claims['Predicted_Claim_Value'] = regressor.predict(claims['Age'].values.reshape(-1, 1)).copy()


# Add a scatterplot of the new column against the age column to the previous scatterplot.
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Claim_Value', data=claims, label='Actual Claim Value')
sns.scatterplot(x='Age', y='Predicted_Claim_Value', data=claims, label='Predicted Claim Value')
plt.title("Actual vs Predicted Claim Value against Age")
plt.xlabel("Age")
plt.ylabel("Claim Value")
plt.legend()
plt.show()

