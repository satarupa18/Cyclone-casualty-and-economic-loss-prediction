import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Load the merged data from the CSV file
data = pd.read_csv('Final_Synthetic_Data.csv')

# Define independent variables for CASUALITIES prediction
independent_vars_casualties = ['Area', 'WIND_SPEED', 'Total_Population', 'Households']

# Define independent variables for economic_loss prediction
independent_vars_economic_loss = ['WIND_SPEED','GDP_VALUE','Total_Population','Males_Population', 'Females_Population']

# Split the data into training and testing sets for CASUALITIES prediction
X_train_casualties, X_test_casualties, y_train_casualties, y_test_casualties = train_test_split(
    data[independent_vars_casualties], data['CASUALITIES'], test_size=0.2, random_state=42
)

# Split the data into training and testing sets for economic_loss prediction
X_train_economic_loss, X_test_economic_loss, y_train_economic_loss, y_test_economic_loss = train_test_split(
    data[independent_vars_economic_loss], data['ECONOMIC_LOSS'], test_size=0.2, random_state=42
)

# Create and fit the Linear Regression model for CASUALITIES prediction
lr_model_casualties = LinearRegression()
lr_model_casualties.fit(X_train_casualties, y_train_casualties)

# Create and fit the Linear Regression model for economic_loss prediction
lr_model_economic_loss = LinearRegression()
lr_model_economic_loss.fit(X_train_economic_loss, y_train_economic_loss)

# Predictions on training and testing data for CASUALITIES prediction
y_train_pred_lr_casualties = lr_model_casualties.predict(X_train_casualties)
y_test_pred_lr_casualties = lr_model_casualties.predict(X_test_casualties)

# Predictions on training and testing data for economic_loss prediction
y_train_pred_lr_economic_loss = lr_model_economic_loss.predict(X_train_economic_loss)
y_test_pred_lr_economic_loss = lr_model_economic_loss.predict(X_test_economic_loss)

# Calculate R-squared and RMSE for CASUALITIES prediction
r2_train_lr_casualties = r2_score(y_train_casualties, y_train_pred_lr_casualties) * 100
r2_test_lr_casualties = r2_score(y_test_casualties, y_test_pred_lr_casualties) * 100
rmse_train_lr_casualties = np.sqrt(mean_squared_error(y_train_casualties, y_train_pred_lr_casualties))
rmse_test_lr_casualties = np.sqrt(mean_squared_error(y_test_casualties, y_test_pred_lr_casualties))

# Calculate R-squared and RMSE for economic_loss prediction
r2_train_lr_economic_loss = r2_score(y_train_economic_loss, y_train_pred_lr_economic_loss) * 100
r2_test_lr_economic_loss = r2_score(y_test_economic_loss, y_test_pred_lr_economic_loss) * 100
rmse_train_lr_economic_loss = np.sqrt(mean_squared_error(y_train_economic_loss, y_train_pred_lr_economic_loss))
rmse_test_lr_economic_loss = np.sqrt(mean_squared_error(y_test_economic_loss, y_test_pred_lr_economic_loss))

# Print evaluation metrics for CASUALITIES prediction
print("Linear Regression Metrics - CASUALITIES:")
print(f"R-squared (training): {r2_train_lr_casualties} %")
print(f"R-squared (testing): {r2_test_lr_casualties} %")
print(f"RMSE (training): {rmse_train_lr_casualties}")
print(f"RMSE (testing): {rmse_test_lr_casualties}")

# Print evaluation metrics for economic_loss prediction
print("\nLinear Regression Metrics - economic_loss:")
print(f"R-squared (training): {r2_train_lr_economic_loss} %")
print(f"R-squared (testing): {r2_test_lr_economic_loss} %")
print(f"RMSE (training): {rmse_train_lr_economic_loss}")
print(f"RMSE (testing): {rmse_test_lr_economic_loss}")
