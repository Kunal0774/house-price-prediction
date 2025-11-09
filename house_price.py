# House Price Prediction Project
# Author: Kunal Chindarkar

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Step 1: Load Dataset
data = pd.read_csv("data.csv")
print("Sample Data:")
print(data.head())

# Step 2: Define features (X) and target (y)
X = data[['area', 'bedrooms', 'bathrooms']]  # input features
y = data['price']  # output label

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create a Linear Regression model
model = LinearRegression()

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Test the model and print accuracy
score = model.score(X_test, y_test)
print(f"\nModel Accuracy: {score * 100:.2f}%")

# Step 7: Predict price for a new house
# Example: 2000 sq ft, 3 bedrooms, 2 bathrooms
new_house = [[2000, 3, 2]]
predicted_price = model.predict(new_house)
print(f"\nPredicted Price for {new_house[0][0]} sq.ft, {new_house[0][1]} BHK, {new_house[0][2]} Bath = ${predicted_price[0]:,.2f}")

# Step 8: Visualization
plt.scatter(data['area'], data['price'], color='blue')
plt.xlabel("Area (sq ft)")
plt.ylabel("Price ($)")
plt.title("House Price vs Area")
plt.show()
