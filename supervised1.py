import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True, as_frame=True)

# Use only the BMI feature
diabetes_X = diabetes_X.loc[:, ['bmi']]

# The BMI is zero-centered and normalized; we recenter it for ease of presentation
diabetes_X = diabetes_X * 30 + 25

# Collect 20 data points
diabetes_X_train = diabetes_X.iloc[-20:]
diabetes_y_train = diabetes_y.iloc[-20:]

# Collect 3 data points
diabetes_X_test = diabetes_X.iloc[:3]
diabetes_y_test = diabetes_y.iloc[:3]

# Display some of the data points
concatingen = pd.concat([diabetes_X_train, diabetes_y_train], axis=1).head()

print(concatingen)


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train.values)

# Make predictions on the training set
diabetes_y_train_pred = regr.predict(diabetes_X_train)

# generate predictions on the new patients
diabetes_y_test_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Slope (theta1): \t', regr.coef_[0])
print('Intercept (theta0): \t', regr.intercept_)

# Plotting
plt.rcParams['figure.figsize'] = [12, 4]

plt.scatter(diabetes_X_train, diabetes_y_train,  color='darkorange')
plt.scatter(diabetes_X_test, diabetes_y_test,  color='blue')
plt.xlabel('Body Mass Index (BMI)')
plt.ylabel('Diabetes Risk')

plt.plot(diabetes_X_test, diabetes_y_test_pred, 'x', color='red', mew=3, markersize=8)
plt.plot(diabetes_X_train, diabetes_y_train_pred, color='black', linewidth=2)

plt.legend(['Model', 'Prediction', 'Initial patients', 'New patients'])

# plt.figure(2)
# theta_list = [(1, 2), (2,1), (1,0), (0,1)]
# for theta0, theta1 in theta_list:
#     x = np.arange(10)
#     y = theta1 * x + theta0
#     plt.plot(x,y)

plt.show()