import math
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error



# London's Latitude and Longitude
LONDON = (51.508045, -0.128217)
LAT = 0
LON = 1

def convert(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points using the Haversine formula
    specified in decimal degrees using atan2
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])


    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    # radius of Earth in kilometers is 6371
    r = 6371.0

    # return result in kilometers
    return c * r



# Coordinates of Location X and Location Y
distance = convert(LONDON[LAT], LONDON[LON], 40.712776, -74.005974)
print("Distance between Location X (London) and Location Y:", distance, " km")

# load customers.xlsx into a DataFrame called data
data = pd.read_excel('customers.xlsx')
# Create a new Series using a numpy array full of zeros (of type float) and the index set data.index from the dataframe just loaded
data['distance'] = pd.Series([0.0]*len(data), index = data.index)
# Copy the 'distance' Series and insert them as columns 8, 9, and 10
data.insert(8, "Latitude", data['distance'].copy())
data.insert(9, "Longitude", data['distance'].copy())
data.insert(10, "Distance", data['distance'].copy())


tosort = [[i, data['Postcode'][i]] for i in range(len(data))]


tosort.sort(key=lambda x: x[1])



nonempty = True
with open('postcodes.csv', 'r') as file:
    postcodes = csv.reader(file)
    for row in postcodes:  
        test = True
        while test and nonempty:
            if row[0] == tosort[0][1]:                                
                data.loc[tosort[0][0], 'Latitude'] = float(row[1])
                data.loc[tosort[0][0], 'Longitude'] = float(row[2])

                # Calculate the distance between the customer's location and London
                data.loc[tosort[0][0], 'Distance'] = convert(LONDON[LAT], LONDON[LON], data.loc[tosort[0][0], 'Latitude'], data.loc[tosort[0][0], 'Longitude'])                                  
                print("ToSort[0]", tosort[0])
                del tosort[0]
                print("ToSort[0]", len(tosort))
                if len(tosort) == 0:   
                    nonempty = False
            else:
                test = False


# Make a new dataframe called claims by using a mask to only copy the rows where there is a Y in the Claim_Made column
claims = data[data['Claim_Made'] == 'Y']

# Valid latitude and longitude values are between 49 and 61 and between -8 and 2
valid_claims = claims[(claims['Latitude'] >= 49) & (claims['Latitude'] <= 61) & (claims['Longitude'] >= -8) & (claims['Longitude'] <= 2)]

print(valid_claims)


# Step 3

x = valid_claims['Distance'].values.reshape(-1, 1)
y = valid_claims['Claim_Value'].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# f. Use a DecisionTreeRegressor to fit the training data. Start with max_depth=3.
regressor = DecisionTreeRegressor(max_depth=3)
regressor.fit(x_train, y_train)

# g. Plot your decision tree
plt.figure(figsize=(12, 8))
plot_tree(regressor, filled=True, feature_names=['Distance'])
plt.title("Decision Tree Regression")
plt.show()

# h. Calculate the average error of predictions compared to the actual test values.
y_pred = regressor.predict(x_test)
error = mean_absolute_error(y_test, y_pred)
print(f"Average error of predictions compared to actual test values: {error}")

# i. Plot the scatterplot of Claim_Value against Distance.
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Distance', y='Claim_Value', data=valid_claims)
plt.title("Scatterplot of Claim_Value against Age")
plt.xlabel("Distance")
plt.ylabel("Claim Value")
plt.show()

# Add a column to the valid claims dataframe of the predicted values from your regressor for all customers.
# valid_claims['Predicted_Claim_Value'] = regressor.predict(valid_claims['Distance'].values.reshape(-1, 1)).copy()

valid_claims.insert(11, "Predicted_Claim_Value", regressor.predict(valid_claims['Distance'].values.reshape(-1, 1)).copy())



# k. Add a scatterplot of the new column against the age column to the previous scatterplot.
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Distance', y='Claim_Value', data=valid_claims, label='Actual Claim Value')
sns.scatterplot(x='Distance', y='Predicted_Claim_Value', data=valid_claims, label='Predicted Claim Value')
plt.title("Actual vs Predicted Claim Value against Distance")
plt.xlabel("Distance")
plt.ylabel("Claim Value")
plt.legend()
plt.show()


