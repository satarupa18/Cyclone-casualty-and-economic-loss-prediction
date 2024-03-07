import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Load the merged data from the CSV file
data = pd.read_csv('Final_Data.csv')

# Define independent variables for CASUALITIES prediction
independent_vars_casualties = [ 'WIND_SPEED', 'Total_Population', 'Males_Population', 'Females_Population', 'Households']

# Split the data into training and testing sets for CASUALITIES
X_train_casualties, X_test_casualties, y_train_casualties, y_test_casualties = train_test_split(
    data[independent_vars_casualties], data['CASUALITIES'], test_size=0.2, random_state=42
)

# Create and fit the Random Forest regressor for CASUALITIES prediction
rf_regressor_casualties = RandomForestRegressor(n_estimators=40, max_depth=7, min_samples_split=2, 
                                                min_samples_leaf=1, max_features='sqrt', random_state=42)
rf_regressor_casualties.fit(X_train_casualties, y_train_casualties)



# Define independent variables for Economic_Loss prediction
independent_vars_economic_loss = ['WIND_SPEED','GDP_VALUE','Inhabited_villages','Towns']


# Split the data into training and testing sets for Economic_Loss
X_train_economic_loss, X_test_economic_loss, y_train_economic_loss, y_test_economic_loss = train_test_split(
    data[independent_vars_economic_loss], data['ECONOMIC_LOSS'], test_size=0.2, random_state=42
)

# Create and fit the Random Forest regressor for Economic_Loss prediction
rf_regressor_economic_loss = RandomForestRegressor(n_estimators=20, max_depth=8, min_samples_split=3, 
                                                   min_samples_leaf=1, max_features='sqrt', random_state=42)
rf_regressor_economic_loss.fit(X_train_economic_loss, y_train_economic_loss)

# Predictions on training and testing data for CASUALITIES
y_train_pred_casualties = rf_regressor_casualties.predict(X_train_casualties)
y_test_pred_casualties = rf_regressor_casualties.predict(X_test_casualties)

# Predictions on training and testing data for Economic_Loss
y_train_pred_economic_loss = rf_regressor_economic_loss.predict(X_train_economic_loss)
y_test_pred_economic_loss = rf_regressor_economic_loss.predict(X_test_economic_loss)

# Calculate R-squared and RMSE for CASUALITIES on training and testing data
r2_train_casualties = r2_score(y_train_casualties, y_train_pred_casualties) * 100
r2_test_casualties = r2_score(y_test_casualties, y_test_pred_casualties) * 100
rmse_train_casualties = np.sqrt(mean_squared_error(y_train_casualties, y_train_pred_casualties))
rmse_test_casualties = np.sqrt(mean_squared_error(y_test_casualties, y_test_pred_casualties))

# Calculate R-squared and RMSE for Economic_Loss on training and testing data
r2_train_economic_loss = r2_score(y_train_economic_loss, y_train_pred_economic_loss) * 100
r2_test_economic_loss = r2_score(y_test_economic_loss, y_test_pred_economic_loss) * 100

rmse_train_economic_loss = np.sqrt(mean_squared_error(y_train_economic_loss, y_train_pred_economic_loss))
rmse_test_economic_loss = np.sqrt(mean_squared_error(y_test_economic_loss, y_test_pred_economic_loss))

#Print evaluation metrics for CASUALITIES
  
print("CASUALITIES Prediction Metrics:")
print(f"R-squared (training): {r2_train_casualties}%")
print(f"R-squared (testing): {r2_test_casualties}%")
print(f"RMSE (training): {rmse_train_casualties}")
print(f"RMSE (testing): {rmse_test_casualties}")

# Print evaluation metrics for Economic_Loss
print("\nEconomic_Loss Prediction Metrics:")
print(f"R-squared (training): {r2_train_economic_loss}%")
print(f"R-squared (testing): {r2_test_economic_loss}%")
print(f"RMSE (training): {rmse_train_economic_loss}")
print(f"RMSE (testing): {rmse_test_economic_loss}")
